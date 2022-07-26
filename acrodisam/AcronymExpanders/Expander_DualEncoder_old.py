'''

@author: jpereira
'''
import random
from Logger import logging

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import numpy as np
import pickle
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Multiply
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate
import tensorflow as tf

import argparse
import numpy as np

from helper import getAcronymChoices, groupby_indexes_unsorted, get_expansion_without_spaces, zip_with_scalar, TrainInstance

from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.AcronymExpander import PredictiveExpander, FactoryPredictiveExpander
import itertools

logger = logging.getLogger(__name__)


def preProcessArticle(acroInstance, trainArticlesDB):
        
    if isinstance(acroInstance, TrainInstance):
        term = get_expansion_without_spaces(acroInstance.expansion)

    else:
        term = acroInstance.acronym
        
    text = acroInstance.getText(trainArticlesDB)
    return text.replace(term, "")

class Factory_Expander_DualEncoder(FactoryPredictiveExpander):

    def __init__(self, maxSeqLen=160):
        FactoryPredictiveExpander.__init__(self)
        random.seed(1337)
        
        try:
            self.maxSeqLen = int(maxSeqLen)
        except ValueError:
            self.maxSeqLen = None
            logger.info("Set maxSeqLen to None, original value cannot be converted to int: " + maxSeqLen)
        
        

        
        #self.loss = args[0]
        #self.c = float(args[1])
                
        #train_c, train_r, train_l = pickle.load(open("/home/jpereira/tmp/dataset-dualencoder-lstm/dataset/" + 'train.pkl', 'rb'))
        #print("ddd")
        
    def getDataPairs(self, y_train):
        positivePairs = set()
        
        for y, indexList in groupby_indexes_unsorted(y_train):
                for t in itertools.combinations(indexList, 2):
                    positivePairs.add(t)
                                
        allPairs = set(itertools.combinations(range(len(y_train)), 2))
        negativePairs = allPairs - positivePairs
            
        if not len(positivePairs) == len(negativePairs):
            if len(positivePairs) < len(negativePairs):
                negativePairs = random.sample(negativePairs, k=len(positivePairs))
            else:
                positivePairs = random.sample(positivePairs, k=len(negativePairs))
                    
        return positivePairs, negativePairs
    
    def getAcronymTrainData(self, trainArticlesDB, X_train, positivePairs, negativePairs):
        leftTrainData = []
        rightTrainData = []
        trainLabels = []
        
        preProcessed = dict()
        for pair, label in itertools.chain(zip_with_scalar(positivePairs,1), zip_with_scalar(negativePairs,0)):
            trainInstance1 = preProcessed.get(pair[0])
            if not trainInstance1:
                trainInstance1 = preProcessArticle(X_train[pair[0]], trainArticlesDB)
                
            trainInstance2 = preProcessed.get(pair[1])
            if not trainInstance2:
                trainInstance2 = preProcessArticle(X_train[pair[1]], trainArticlesDB)
                
            leftTrainData.append(trainInstance1)
            rightTrainData.append(trainInstance2)
            trainLabels.append(label)
            
        return leftTrainData, rightTrainData, trainLabels
        
    def getTrainData(self, trainArticlesDB, acronymDB, fold):
        #TODO load if exits use dataset and fold
        
        leftTrainData = []
        rightTrainData = []
        trainLabels = []
        # Build train data, an acronym at a time
        for acronym in acronymDB.keys():
            X_train, y_train, labelToExpansion = getAcronymChoices(acronym, acronymDB)
            
            positivePairs, negativePairs = self.getDataPairs(y_train)
            acroLeftTrainData, acroRightTrainData, acroTrainLabels = self.getAcronymTrainData(trainArticlesDB, X_train, positivePairs, negativePairs)
            
            leftTrainData.extend(acroLeftTrainData)
            rightTrainData.extend(acroRightTrainData)
            trainLabels.extend(acroTrainLabels)
            
            
        # Transform Data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(leftTrainData)
        tokenizer.fit_on_texts(rightTrainData)
    
        train_c = tokenizer.texts_to_sequences(leftTrainData)
        train_r = tokenizer.texts_to_sequences(rightTrainData)
    
    
        if not self.maxSeqLen or self.maxSeqLen < 1:
            maxSeqLen = max([len(seq) for seq in train_c + train_r])
            logger.info("No MaxSeqLen, set to maximum: " + str(maxSeqLen))
        else:
            maxSeqLen = self.maxSeqLen
             
        #MAX_SEQUENCE_LENGTH = max([len(seq) for seq in train_c + train_r
                                                        #+ test_c + test_r
                                                        #+ dev_c + dev_r])
        
        max_number_words = len(tokenizer.word_index) + 1
        word_index = tokenizer.word_index
        
        
        logger.info("MAX_NB_WORDS: {}".format(max_number_words))
    
        train_c = pad_sequences(train_c, maxlen=maxSeqLen)
        train_r = pad_sequences(train_r, maxlen=maxSeqLen)
    
        # shuffle training set
        indices = np.arange(train_c.shape[0])
        
        np.random.shuffle(indices)
        train_l = trainLabels
    
        train_c = np.asarray(train_c)
        train_r = np.asarray(train_r)
        train_l = np.asarray(train_l)
    
        train_c = train_c[indices]
        train_r = train_r[indices]
        train_l = train_l[indices]        
        
        return train_c, train_r, train_l, tokenizer, maxSeqLen, max_number_words, word_index
        #TODO save into disk
        #pickle.dump([train_c, train_r, train_l], open(DATA_DIR + "train.pkl", "wb"), protocol=-1)
        #pickle.dump([test_c, test_r, test_l], open(DATA_DIR + "test.pkl", "wb"), protocol=-1)
        #pickle.dump([dev_c, dev_r, dev_l], open(DATA_DIR + "dev.pkl", "wb"), protocol=-1)

        #pickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open(DATA_DIR + "params.pkl", "wb"), protocol=-1)
    
    def getDualEncoder(self, train_c, train_r, train_l, maxSeqLen, maxNWords, word_index):
        
        
        '''
        parser = argparse.ArgumentParser()
        parser.register('type','bool',self.str2bool)
        parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
        parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
        parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
        parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
        parser.add_argument('--input_dir', type=str, default='./dataset/', help='Input dir')
        parser.add_argument('--save_model', type='bool', default=True, help='Whether to save the model')
        parser.add_argument('--model_fname', type=str, default='model/dual_encoder_lstm_classifier.h5', help='Model filename')
        parser.add_argument('--embedding_file', type=str, default='embeddings/glove.840B.300d.txt', help='Embedding filename')
        parser.add_argument('--seed', type=int, default=1337, help='Random seed')
        args = parser.parse_args(args=None, namespace=None)
        print ('Model args: ', args)
        np.random.seed(args.seed)
        '''
        logger.info("Starting...")
        
        # first, build index mapping words in the embeddings set
        # to their embedding vector
        embedding_file = "/home/jpereira/git/AcroDisam/acrodisam_app/data/GloveEmbeddings/glove.840B.300d.txt"
        emb_dim = 300
        hidden_size = 300
        optimizer = "adam"
        batch_size = 256
        #n_epochs = 50
        n_epochs = 20
        
        
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
        num_words = min(maxNWords, len(word_index)) + 1
        embedding_matrix = np.zeros((num_words , emb_dim))
        for word, i in word_index.items():
            if i >= maxNWords:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        
        
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
        #with tf.device('/cpu:0'):
        # define lstm encoder
        encoder = Sequential()
        encoder.add(Embedding(output_dim=emb_dim,
                                input_dim=maxNWords,
                                input_length=maxSeqLen,
                                weights=[embedding_matrix],
                                mask_zero=True,
                                trainable=True))
        
        encoder.add(LSTM(units=hidden_size))
        
        context_input = Input(shape=(maxSeqLen,), dtype='int32')
        response_input = Input(shape=(maxSeqLen,), dtype='int32')
    
        # encode the context and the response
        context_branch = encoder(context_input)
        response_branch = encoder(response_input)
        
        #concatenated = concatenate([context_branch, response_branch], mode='mul')
        concatenated = Multiply()([context_branch, response_branch])
        out = Dense((1), activation = "sigmoid") (concatenated)
    
        dual_encoder = Model([context_input, response_input], out)
        dual_encoder.compile(loss='binary_crossentropy',
                        optimizer=optimizer)
        
        logger.info(encoder.summary())
        logger.info(dual_encoder.summary())
        
        
        logger.info('Found %s training samples.' % len(train_c))
        
        #histories = my_callbacks.Histories()
        
        #bestAcc = 0.0
        #patience = 0 
        
        logger.info("\tbatch_size={}, nb_epoch={}".format(batch_size, n_epochs))
        
        dual_encoder.fit([train_c, train_r], train_l,
                    batch_size=batch_size, epochs=n_epochs, #callbacks=[histories],
                    #validation_data=([dev_c, dev_r], dev_l),
                     verbose=1)
        """
        for ep in range(1, n_epochs):
                    
            dual_encoder.fit([train_c, train_r], train_l,
                    batch_size=batch_size, epochs=1, #callbacks=[histories],
                    #validation_data=([dev_c, dev_r], dev_l),
                     verbose=1)
        """
        #if save_model:
        #    print("Now saving the model... at {}".format(args.model_fname))
        #    dual_encoder.save(args.model_fname)
        
        return dual_encoder
    
            
        
    
    

        
    def getExpander(self, trainArticlesDB, acronymDB = None, articleAcronymDB = None, fold="", executionTimeObserver = None):
        train_c, train_r, train_l, tokenizer, maxSeqLen, maxNWords, word_index = self.getTrainData(trainArticlesDB, acronymDB, fold)
        #textRepresentator = self.getRepresentator(trainArticlesDB = trainArticlesDB,
        #                                          articleAcronymDB = articleAcronymDB,
        #                                          fold = fold,
        #                                          executionTimeObserver=executionTimeObserver)
        
        dualEconder = self.getDualEncoder(train_c, train_r, train_l, maxSeqLen, maxNWords, word_index)
        return Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)


class Expander_DualEncoder(PredictiveExpander):

    def __init__(self, dualEconder, articlesDB, tokenizer, maxSeqLen, expander_type=AcronymExpanderEnum.none):
        PredictiveExpander.__init__(self, expander_type, None)
        self.dualEncoder = dualEconder
        self.articlesDB = articlesDB
        self.tokenizer = tokenizer
        self.maxSeqLen = maxSeqLen
        
    def transform(self, X):
        X = super().transform(X)
        text = [preProcessArticle(x, self.articlesDB) for x in X]
        tokens = self.tokenizer.texts_to_sequences(text)
        seq = pad_sequences(tokens, maxlen=self.maxSeqLen)
        return seq
    
    def fit(self, X_train, y_train):
        
        self.X_train = X_train
        self.y_train = y_train        
        
        '''
        distinctLabels = set(y_train)
        num_classes = self.num_classes = len(distinctLabels)
        
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_shape=X_train[0].shape))
        self.model.add(Dense(units=num_classes, activation='softmax'))
                
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        one_hot_labels = to_categorical(y_train, num_classes=num_classes)
        
        x_matrix = np.array(X_train)
        
        self.x_matrix = x_matrix
        
        self.model.fit(x_matrix, one_hot_labels, epochs=5, batch_size=32)
        '''
        
    def predict(self, X_test, acronym):
        
        labels = []
        confidences = []
        
        for x_test in X_test:            
            # DO repeat
            repeated_x_test = np.tile(x_test, (len(self.X_train),1))
            probablities = self.dualEncoder.predict([self.X_train, repeated_x_test])   
        
            mostSimilarX =  np.argmax(probablities, axis=0)[0]
            labels.append(self.y_train[mostSimilarX])
            confidences.append(probablities[mostSimilarX])
        
        
        
    
#         for x in X_test:
#             y = self.model.predict(np.array(x))[0]
# 
#             label = np.argmax(y)
#             confidence = y[label]
#             
#             labels.append(label)
#             confidences.append(confidence)
    #    x_matrix = np.array(X_test)
    #    labels = self.model.predict(x_matrix, batch_size=128)
    #    if self.num_classes > 2:
    #        print("HEre")
        
        #labels = self.classifier.predict(X_test)
        
        #decisions = self.classifier.decision_function(X_test)
        
        #confidences = self._getConfidencesFromDecisionFunction(labels, decisions)
        
        return labels,  confidences
