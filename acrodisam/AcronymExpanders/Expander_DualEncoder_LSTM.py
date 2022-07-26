'''

@author: jpereira
'''
from Logger import logging

import numpy as np
from keras.layers import Input, LSTM, Dense, Embedding, Multiply
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import text_preparation
from AcronymExpanders.Expander_Generic_LearnSim import Factory_Expander_Generic_LearnSim
from string_constants import FILE_LSTM_DUAL_ENCODER, FILE_GLOVE_EMBEDDINGS


logger = logging.getLogger(__name__)



    

class Factory_Expander_DualEncoder_LSTM(Factory_Expander_Generic_LearnSim):

    def __init__(self, maxSeqLen=160, n_epoch = 30, datasetName=None, saveAndLoad=False, persistentArticles=None):
        Factory_Expander_Generic_LearnSim.__init__(self, maxSeqLen, n_epoch, datasetName, saveAndLoad, persistentArticles)
        
        self.embeddings = "glove"
        self.hidden_size = 300        
        self.optimizer = "adam"
        self.batch_size = 256
    
    def preProcessInstance(self, acroInstance, trainArticlesDB, firstSentence= True, numSeq = None):
        return text_preparation.preProcessArticle(acroInstance,trainArticlesDB, numSeq)
    
    def preProcessRunningInstance(self, tokenizer, acroInstance, trainArticlesDB, firstSentence= True, numSeq = None):

        tokens = text_preparation.preProcessArticle(acroInstance, trainArticlesDB, numSeq, lambda text: tokenizer.texts_to_sequences([text])[0])
        seq = pad_sequences([tokens], maxlen=self.maxSeqLen)
        return seq[0]

    
    def getModel(self, leftTrainData, rightTrainData, trainLabels, maxSeqLen, fold, executionTimeObserver=None):
        
        tokenizer = Tokenizer(filters='')
        
        left_list = leftTrainData.tolist()
        right_list = rightTrainData.tolist()
        
        tokenizer.fit_on_texts(left_list)
        tokenizer.fit_on_texts(right_list)
    
        preProcessRunningInstance = lambda acroInstance, trainArticlesDB, firstSentence, numSeq: self.preProcessRunningInstance(tokenizer, acroInstance, trainArticlesDB, firstSentence, numSeq)
        
        train_c = tokenizer.texts_to_sequences(left_list)
        train_r = tokenizer.texts_to_sequences(right_list)
        
        max_number_words = len(tokenizer.word_index) + 1
        word_index = tokenizer.word_index
        
        
        logger.info("MAX_NB_WORDS: {}".format(max_number_words))
    
        train_c = pad_sequences(train_c, maxlen=maxSeqLen)
        train_r = pad_sequences(train_r, maxlen=maxSeqLen)
        train_l = trainLabels
        
        logger.info("Starting...")

        # first, build index mapping words in the embeddings set
        # to their embedding vector
        if self.embeddings == "glove":
            embedding_file = FILE_GLOVE_EMBEDDINGS
            emb_dim = 300
            
            logger.info('Now indexing word vectors...')
        
            embeddings_index = {}
            f = open(embedding_file, 'r')
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                except ValueError:
                    continue
                embeddings_index[word] = coefs
            f.close()
            
            #MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index = pickle.load(open(args.input_dir + 'params.pkl', 'rb'))
        
            
            logger.info("Now loading embedding matrix...")
            num_words = min(max_number_words, len(word_index)) + 1
            embedding_matrix = np.zeros((num_words , emb_dim))
            for word, i in word_index.items():
                if i >= max_number_words:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            
            weights = [embedding_matrix]
        
        #TODO
        # if Word2vec embeddings
        # init embedding size test num:
        
        else:
            weights = None
            emb_dim = 300
        

        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Create 2 virtual GPUs with 1GB memory each
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
                    #"tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                logger.error(e)
        
        """
        logger.info("Now creating the model...")

        encoder = Sequential()
        encoder.add(Embedding(output_dim=emb_dim,
                                input_dim=max_number_words,
                                input_length=maxSeqLen,
                                weights=weights,
                                mask_zero=True,
                                trainable=True))
        
        encoder.add(LSTM(units=self.hidden_size))
        
        context_input = Input(shape=(maxSeqLen,), dtype='int32')
        response_input = Input(shape=(maxSeqLen,), dtype='int32')
    
        # encode the context and the response
        context_branch = encoder(context_input)
        response_branch = encoder(response_input)
        
        #replaces concatenated = concatenate([context_branch, response_branch], mode='mul')
        concatenated = Multiply()([context_branch, response_branch])
        out = Dense((1), activation = "sigmoid") (concatenated)
    
        dual_encoder = Model([context_input, response_input], out)
        dual_encoder.compile(loss='binary_crossentropy',
                        optimizer=self.optimizer)
        
        logger.info(encoder.summary())
        logger.info(dual_encoder.summary())
        
        
        logger.info('Found %s training samples.' % len(train_c))
        
        logger.info("\tbatch_size={}, nb_epoch={}".format(self.batch_size, self.n_epoch))
        if executionTimeObserver:
            executionTimeObserver.start()
        dual_encoder.fit([train_c, train_r], train_l,
                    batch_size=self.batch_size, epochs=self.n_epoch, 
                     verbose=1)
        if executionTimeObserver:
            executionTimeObserver.stop()

        #if save_model:
        #    print("Now saving the model... at {}".format(args.model_fname))
        #    dual_encoder.save(args.model_fname)
        model = lambda X_train, x_test : predict(X_train, x_test, dual_encoder)
        
        return model, preProcessRunningInstance
    
    

        
#TODO model save and load         
#         if self.saveAndLoad:
#             generatedFilesFolder = getDatasetGeneratedFilesPath(self.datasetName)
#             varToName = "_".join([str(s) for s in [self.datasetName,fold, self.maxSeqLen, self.embeddings, self.hidden_size, self.n_epochs]])
#             dualEncoderLocation = generatedFilesFolder + FILE_LSTM_DUAL_ENCODER + "_" + varToName + ".h5"
#                 
#             if os.path.isfile(dualEncoderLocation):
#                 # TODO Load Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
#                 return load_model(dualEncoderLocation)    
#  
#         
#         train_c, train_r, train_l, tokenizer, maxSeqLen, maxNWords, word_index = self.getTrainData(trainArticlesDB, 
#                                                                                                    acronymDB,
#                                                                                                    datasetName = self.datasetName, 
#                                                                                                    fold = fold, 
#                                                                                                    persistentArticles = self.persistentArticles)
#         #textRepresentator = self.getRepresentator(trainArticlesDB = trainArticlesDB,
#         #                                          articleAcronymDB = articleAcronymDB,
#         #                                          fold = fold,
#         #                                          executionTimeObserver=executionTimeObserver)
#         
#         dualEconder = self.getDualEncoder(train_c, train_r, train_l, maxSeqLen, maxNWords, word_index, 
#                                           embeddings=self.embeddings, hidden_size = self.hidden_size, n_epochs = self.n_epochs, 
#                                           executionTimeObserver=executionTimeObserver)
#         
#         if self.saveAndLoad:
#             # TODO save Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
#             dualEconder.save(dualEncoderLocation)
#         


def predict(X_train, x_test, dualEncoder):
             
    repeated_x_test = np.tile(x_test, (len(X_train),1))
    probablities = dualEncoder.predict([X_train, repeated_x_test])   
    return [pro[0] for pro in probablities]

