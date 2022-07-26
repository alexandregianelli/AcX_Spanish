'''
Selects the expansion whose vector obtains the closer cosine distance to the test instance
'''
from scipy.sparse import csr_matrix, vstack
import numpy as np
from out_expanders.impl.cossim import _ExpanderCossimText
from text_representators.impl.context_vector import FactoryContextVector
from helper import TrainInstance
from acronym_expander import RunConfig
from .._base import OutExpanderWithTextRepresentator, OutExpanderWithTextRepresentatorFactory


class FactoryClassicContextVector(OutExpanderWithTextRepresentatorFactory):

                
    def get_expander(self, articles_db, acronym_db=None, article_acronym_db=None, fold="", execution_time_observer=None):
        
        text_representator = self._get_representator(articles_db=articles_db,
                                                  article_acronym_db=article_acronym_db,
                                                  fold=fold,
                                                  execution_time_observer=execution_time_observer)
        return ExpanderClassicContextVector(text_representator)


class ExpanderClassicContextVector(_ExpanderCossimText):

    def __init__(self, text_representator):
        super().__init__(text_representator = text_representator)


    def fit(self, X_train, y_train):
        self.X_train = []
        self.y_train = []
        
        for i in range(0, len(y_train)):
            y = y_train[i]
            if y not in self.y_train:
                self.y_train.append(y)
                self.X_train.append(X_train[i])
                
        self.X_train = vstack(self.X_train)
        
    def transform(self, X):
        expansionsDict = {}
        for x in X:
            if isinstance(x, TrainInstance):
                expansion = x.expansion
                if expansion not in expansionsDict:
                    expansionsDict[expansion] = []
                expansionsDict[expansion].append(x)
            else:
                return vstack([self.text_representator.transformInstance(x) for x in X])
        
        expansionsEmbeddings = {}
        for expansion, listX in expansionsDict.items():
            expansionsEmbeddings[expansion] = csr_matrix((1,self.text_representator.vector_size), dtype=np.float64)
            for x in listX:
                expansionsEmbeddings[expansion] += self.text_representator.transformInstance(x)
                
        #return expansionsEmbeddings 
        return [expansionsEmbeddings[x.expansion] for x in X]
