'''

@author: jpereira
'''


import random
from Logger import logging

from tensorflow import keras
from tensorflow.keras.layers import Input, Multiply, Dense
from tensorflow.keras.models import Model

from AcronymExpanders.Expander_ALBERT import Factory_Expander_ALBERT
import AcronymExpanders.Expander_DualEncoder_LSTM
from TextRepresentators.Representator_ALBERT import preProcessInstance

from DataCreators.BertBasedModel import getAlbertModel, loadAlbertWeights
logger = logging.getLogger(__name__)


class Factory_Expander_ALBERTDualEncoder(Factory_Expander_ALBERT):

    def __init__(self, maxSeqLen=160, datasetName=None, saveAndLoad=False, persistentArticles=None):  
        Factory_Expander_ALBERT.__init__(self, maxSeqLen=maxSeqLen, datasetName=datasetName, saveAndLoad=saveAndLoad, persistentArticles=persistentArticles)
        random.seed(1337)

        self.batch_size = 40
        self.n_epochs = 2
        self.optimizer = "adam"

    def preProcessInstance(self, acroInstance, trainArticlesDB, firstSentence= True, numSeq = None):
        
        sentenceProcessor = self.getSentenceProcessor()

        return preProcessInstance(acroInstance, trainArticlesDB, sentenceProcessor, numSeq)

    
    def getModel(self, leftTrainData, rightTrainData, trainLabels, maxSeqLen, fold, executionTimeObserver=None):
        
        logger.info("Starting...")
        
        l_bert = getAlbertModel()

        context_input_ids      = Input(shape=(maxSeqLen,), dtype='int32', name="context_input_ids")
        context_output = l_bert(context_input_ids)
        context_output = keras.layers.Lambda(lambda x: x[:, 0, :])(context_output)
        
        response_input_ids      = Input(shape=(maxSeqLen,), dtype='int32', name="response_input_ids")
        response_output         = l_bert(response_input_ids)
        response_output = keras.layers.Lambda(lambda x: x[:, 0, :])(response_output)
    
        
        logger.info("Now creating the model...")
        
        concatenated = Multiply()([context_output, response_output])
        out = Dense((1), activation = "sigmoid") (concatenated)
    
        dual_encoder = Model([context_input_ids, response_input_ids], out)
        
        dual_encoder.build(input_shape=(None, maxSeqLen))
        
        dual_encoder.compile(loss='binary_crossentropy',
                        optimizer=self.optimizer)
        
        
        
        logger.info(dual_encoder.summary())
        
        loadAlbertWeights(l_bert)
            
        dual_encoder.summary()
        
        logger.info('Found %s training samples.' % len(leftTrainData))
        
        logger.info("\tbatch_size={}, nb_epoch={}".format(self.batch_size, self.n_epochs))
        if executionTimeObserver:
            executionTimeObserver.start()
        dual_encoder.fit([leftTrainData, rightTrainData], trainLabels,
                    batch_size=self.batch_size, epochs=self.n_epochs,
                     verbose=1)
        if executionTimeObserver:
            executionTimeObserver.stop()

        # TODO save and load model
        #if save_model:
        #    print("Now saving the model... at {}".format(args.model_fname))
        #    dual_encoder.save(args.model_fname)
        
        modelP = lambda X_train, x_test : AcronymExpanders.Expander_DualEncoder_LSTM.predict(X_train, x_test, dual_encoder)
        
        return modelP, self.preProcessInstance
    
