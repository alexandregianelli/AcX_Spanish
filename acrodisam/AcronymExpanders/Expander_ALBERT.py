'''

@author: jpereira
'''
import os
import random
from Logger import logging
import numpy as np

import bert
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import text_preparation
from helper import TrainInstance
from AcronymExpanders.Expander_Generic_LearnSim import  Factory_Expander_Generic_LearnSim
from DataCreators.BertBasedModel import getAlbertModel, loadAlbertWeights, getAlbertSentenceProcessor

from helper import getDatasetGeneratedFilesPath
from string_constants import FILE_ALBERT

logger = logging.getLogger(__name__)


class Factory_Expander_ALBERT(Factory_Expander_Generic_LearnSim):

    def __init__(self, preTrainedModel = "albert_base", maxSeqLen=160, n_epoch = 1, datasetName=None, saveAndLoad=False, persistentArticles=None):
        Factory_Expander_Generic_LearnSim.__init__(self, maxSeqLen, n_epoch, datasetName, saveAndLoad, persistentArticles)
        random.seed(1337)
        
        #self.embeddings = 300 
        #self.n_epochs = 1
        self.preTrainedModel = preTrainedModel
        #self.batch_size = 160
        
        # xxlarge gpu
        if preTrainedModel == 'albert_base':
            self.batch_size = 16
        elif preTrainedModel == 'albert_xxlarge':
            self.batch_size = 4
        else:
            self.batch_size = 4

        self.sentenceProcessor = None

                        
    def getSentenceProcessor(self):
        if not self.sentenceProcessor:
            self.sentenceProcessor = getAlbertSentenceProcessor(self.preTrainedModel)
        return self.sentenceProcessor
        
    def preProcessInstance(self, acroInstance, trainArticlesDB, firstSentence= True, numSeq = None):
        sentenceProcessor = self.getSentenceProcessor()
        
        if firstSentence == True or (firstSentence == None and isinstance(acroInstance, TrainInstance)):
            numSeqArg = numSeq//2 - 2
        else:
            numSeqArg = numSeq//2 - 1
    
        tokens = text_preparation.preProcessArticle(acroInstance, 
                                             trainArticlesDB, 
                                             numSeq = numSeqArg, 
                                             tokenizer = lambda text: sentenceProcessor.encode_as_ids(text))
        
        if firstSentence:
            new_tokens = [sentenceProcessor.piece_to_id("[CLS]")] + tokens + [sentenceProcessor.piece_to_id("[SEP]")]
        else:
            new_tokens = tokens + [sentenceProcessor.piece_to_id("[SEP]")]
    
    
        spanId = sentenceProcessor.piece_to_id("<span>")
        seq = pad_sequences([new_tokens], maxlen=numSeq//2 , padding='post',truncating='post', value = spanId)
        
        if len(seq[0]) < len(tokens):
            logger.error("Pre processing failed, len(tokens) is higher than numseq")
            tokens = text_preparation.preProcessArticle(acroInstance, 
                                             trainArticlesDB, 
                                             numSeq = numSeqArg, 
                                             tokenizer = lambda text: sentenceProcessor.encode_as_ids(text))
        
        return seq[0]


    def getModel(self, leftTrainData, rightTrainData, trainLabels, maxSeqLen, fold, executionTimeObserver=None):

            
        if self.saveAndLoad:
            generatedFilesFolder = getDatasetGeneratedFilesPath(self.datasetName)
            varToName = "_".join([str(s) for s in [self.datasetName,fold,self.preTrainedModel,self.maxSeqLen,self.n_epoch,self.batch_size]])

            modelLocation = generatedFilesFolder + FILE_ALBERT + "_" + varToName + ".h5"            
        
            if os.path.isfile(modelLocation):
                model = load_model(modelLocation, custom_objects={"BertModelLayer": bert.BertModelLayer})
            else:
                model = self.trainModel(leftTrainData, rightTrainData, trainLabels, maxSeqLen, executionTimeObserver)
                logger.info("Now saving the model... at {}".format(modelLocation))
                model.save(modelLocation)
        else:
            model = self.trainModel(self, leftTrainData, rightTrainData, trainLabels, maxSeqLen, executionTimeObserver)
        
        modelP = lambda X_train, x_test : predict(X_train, x_test, model, maxSeqLen, self.getSentenceProcessor())
        
        return modelP, self.preProcessInstance
    
            
    def trainModel(self, leftTrainData, rightTrainData, trainLabels, maxSeqLen, executionTimeObserver=None):
        logger.info("Starting...")
        
        l_bert = getAlbertModel(self.preTrainedModel)

        # use in Keras Model here, and call model.build()

        # Uncomment to use CPU
        #with tf.device('/cpu:0'):
        #pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
        #model3  = Model(inputs=model.input, outputs=pool_output)
        #model3.compile(loss='binary_crossentropy', optimizer=adam)
        #model3.summary()
        
        
        context_input_ids      = Input(shape=(maxSeqLen,), dtype='int32', name="context_input_ids")
        context_token_type_ids = Input(shape=(maxSeqLen,), dtype='int32', name="context_token_type_ids")
        context_output         = l_bert([context_input_ids, context_token_type_ids])
        
        # TODO recheck with bert QA
        context_output = keras.layers.Lambda(lambda x: x[:, 0, :])(context_output)
        
        
        logger.info("Now creating the model...")

        out = Dense((1), activation = "sigmoid") (context_output)
    
        model = Model([context_input_ids, context_token_type_ids], out)
        
        model.build(input_shape=[(None, maxSeqLen), (None, maxSeqLen)])
        
        #    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        model.compile(loss='binary_crossentropy',
                        optimizer=keras.optimizers.Adam())
        
        logger.info(model.summary()) 
        
        loadAlbertWeights(l_bert, self.preTrainedModel)
            
        model.summary()
        
        logger.info('Found %s training samples.' % len(trainLabels))
        
        
        train_c = leftTrainData
        train_r = rightTrainData

        train_ids = [np.concatenate([leftText, rightText]) for leftText, rightText in zip(train_c, train_r)]
        train_seq = [[0] * len(leftText) + [1] * len(rightText) for leftText, rightText in zip(train_c, train_r)]

        train_ids = np.array(train_ids,  dtype=np.int32)
        train_seq = np.array(train_seq,  dtype=np.int32)
        
        
        
        logger.info("\tbatch_size={}, nb_epoch={}".format(self.batch_size, self.n_epoch ))
        if executionTimeObserver:
            executionTimeObserver.start()
        model.fit((train_ids, train_seq), trainLabels,
                    batch_size=self.batch_size, epochs=self.n_epoch,
                     verbose=1)
        if executionTimeObserver:
            executionTimeObserver.stop()
    
    
        return model

    """
    def getExpander(self, trainArticlesDB, acronymDB = None, articleAcronymDB = None, fold="", executionTimeObserver = None):
        
        self.sp_model = tf.io.gfile.glob(os.path.join("/home/jpereira/git/AcroDisam/acrodisam_app/data/albert/albert_base", "assets/*"))[0]
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.sp_model)
        
        
        if self.saveAndLoad:
            generatedFilesFolder = getDatasetGeneratedFilesPath(self.datasetName)
            varToName = "_".join([str(s) for s in [self.datasetName,fold, self.maxSeqLen, self.embeddings, self.hidden_size, self.n_epochs]])
            expander_albert_location = generatedFilesFolder + "EXPANDER_ALBERT" + "_" + varToName + ".pickle"
                
            if os.path.isfile(expander_albert_location):
                # TODO Load Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
                model, maxSeqLen = pickle.load(expander_albert_location)
                Expander_ALBERT(model, trainArticlesDB, self.sp , maxSeqLen) 
 
        
        train_ids, train_seq, train_l, maxSeqLen = self.getTrainData(trainArticlesDB, 
                                                                                                   acronymDB,
                                                                                                   datasetName = self.datasetName, 
                                                                                                   fold = fold, 
                                                                                                   persistentArticles = self.persistentArticles)
        #textRepresentator = self.getRepresentator(trainArticlesDB = trainArticlesDB,
        #                                          articleAcronymDB = articleAcronymDB,
        #                                          fold = fold,
        #                                          executionTimeObserver=executionTimeObserver)
        
        model = self.getModel(train_ids, train_seq, train_l, maxSeqLen, 
                                          embeddings=self.embeddings, hidden_size = self.hidden_size, n_epochs = self.n_epochs, 
                                          executionTimeObserver=executionTimeObserver)
        
        if self.saveAndLoad:
            # TODO save Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
            #dualEconder.save(dualEncoderLocation)
            pickle.dump([model, maxSeqLen], open(expander_albert_location, "wb"), protocol=-1)

        
        return Expander_ALBERT(model, trainArticlesDB, self.sp , maxSeqLen)
    """


def albertInputGenerator(X_train, x_test, maxSeqLen, sentenceProcessor):
    testIDs = x_test
    testSeq = [1] * len(testIDs)
    
    for x in X_train:
        
        trainIDs = x
        trainSeq = [0] * len(trainIDs) 
        
        ids = np.concatenate([trainIDs, testIDs])
        seq =  trainSeq + testSeq 
        
        spanId = sentenceProcessor.piece_to_id("<span>")
        ids = pad_sequences([ids], maxlen=maxSeqLen , padding='post',truncating='post', value = spanId)
        seq = pad_sequences([seq], maxlen=maxSeqLen, padding='post',truncating='post', value=1)
        
        yield ids[0], seq[0]

def predict(X_train, x_test, model, maxSeqLen, sentenceProcessor):
    
    probablities = [model.predict((np.array([ids]), np.array([seq])))[0][0] for ids, seq in albertInputGenerator(X_train, x_test, maxSeqLen, sentenceProcessor)]

    return np.asarray(probablities)

