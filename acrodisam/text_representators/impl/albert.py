
from Logger import logging



from helper import TrainInstance
import text_preparation

import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .._base import TextRepresentator, TextRepresentatorFactory
from DataCreators.BertBasedModel import getAlbertModel, loadAlbertWeights, getAlbertSentenceProcessor


logger = logging.getLogger(__name__)



def preProcessInstance(acroInstance, trainArticlesDB, sentenceProcessor, numSeq = None):
                tokens = text_preparation.preProcessArticle(acroInstance, 
                                                     trainArticlesDB, 
                                                     numSeq = numSeq - 2, 
                                                     tokenizer = lambda text: sentenceProcessor.encode_as_ids(text))
                
                tokens = [sentenceProcessor.piece_to_id("[CLS]")] + tokens + [sentenceProcessor.piece_to_id("[SEP]")]

                spanId = sentenceProcessor.piece_to_id("<span>")
                seq = pad_sequences([tokens], maxlen=numSeq, padding='post',truncating='post', value = spanId)
                return seq[0]

class Factory_ALBERT(TextRepresentatorFactor):

    def __init__(self, *args, dataset_name=None, save_and_load=False, persistent_articles=None):

        
        self.datasetName = dataset_name
        self.saveAndLoad = save_and_load
        
        self.persistentArticles = persistent_articles
        
        self.maxSeqLen = int(args[0])
        
        self.fine_tune = True
        
        self.batch_size = 32
        self.epochs = 1
    
    def getTrainData(self, trainArticlesDB, articleAcronymDB, sp):
        articles = []
        
        for article_id, acroExp in articleAcronymDB.items():
            for acronym, expansion in acroExp.items():
                instance = TrainInstance(article_id = article_id, acronym=acronym, expansion=expansion)
                seq = preProcessInstance(instance, trainArticlesDB, sentenceProcessor = sp, numSeq= self.maxSeqLen)
                
                articles.append(seq)
        return articles
    
    
    def buildAlbert(self, trainArticlesDB = None, articleAcronymDB = None, sentenceProcessor = None, executionTimeObserver=None):
        
        logger.info("Starting...")
        
        l_bert = getAlbertModel()
        
        context_input_ids      = Input(shape=(self.maxSeqLen,), dtype='int32', name="context_input_ids")
        context_output = l_bert(context_input_ids)

        if self.fine_tune:
            model = keras.Model(inputs=[context_input_ids], outputs=context_output)
            model.build(input_shape=[(None, self.maxSeqLen)])
    
            loadAlbertWeights(l_bert)

            
            model.compile(optimizer=keras.optimizers.Adam(),
                          loss=keras.losses.mean_squared_error)
            
            input_ids = np.array(self.getTrainData(trainArticlesDB, articleAcronymDB, sentenceProcessor),  dtype=np.int32)
    
            pres = model.predict(input_ids)# just for fetching the shape of the output
            print("pres:", pres.shape)
            logger.info(model.summary())

            if executionTimeObserver:
                executionTimeObserver.start()
                
            model.fit(x=input_ids,
                      y=np.zeros_like(pres),
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      verbose=1)
            
            if executionTimeObserver:
                executionTimeObserver.stop()
            
            
            # TODO check other functions to create embeddings
            context_output = keras.layers.Lambda(lambda x: x[:, 0, :])(context_output)
            model = Model([context_input_ids], context_output)
            
            model.build(input_shape=(None, self.maxSeqLen))
            logger.info(model.summary())

        else:
        
            context_output = keras.layers.Lambda(lambda x: x[:, 0, :])(context_output)
        
            logger.info("Now creating the model...")

            model = Model([context_input_ids], context_output)
            
            model.build(input_shape=(None, self.maxSeqLen))
        
            logger.info(model.summary())
        
            loadAlbertWeights(l_bert)
            
            
        model.summary()

        return model
        
    def get_text_representator(self, train_articles_db, article_acronym_db, fold, execution_time_observer= None):
        
        sp = getAlbertSentenceProcessor()
        
        albertModel = self.buildAlbert(trainArticlesDB = train_articles_db, 
                                       articleAcronymDB = article_acronym_db, 
                                       sentenceProcessor=sp, 
                                       executionTimeObserver = execution_time_observer)
        
        return Representator_ALBERT(albertModel, train_articles_db, sp, self.maxSeqLen)


class Representator_ALBERT(TextRepresentator):
    """
    """

    def __init__(self, albertModel, trainArticlesDB, sp, maxSeqLen):
        super().__init__()
        self.albertModel = albertModel
        self.trainArticlesDB = trainArticlesDB
        self.sp = sp
        self.maxSeqLen = maxSeqLen

    def transform(self, X):
        results = []
        for item in X:
            seq = preProcessInstance(item, self.trainArticlesDB, sentenceProcessor = self.sp, numSeq= self.maxSeqLen)
                
            vect = self.albertModel(np.array([seq]))[0]
            results.append(vect)
        return results

