import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

from TextRepresentators import TextRepresentatorEnum
from TextRepresentators.TextRepresentator import TextRepresentator, TextRepresentatorFactory
from helper import TrainInstance
from text_preparation import get_expansion_without_spaces


class Factory_ConcatRepresentators(TextRepresentatorFactory):

    def __init__(self, representatorFactories):
        self.representatorFactories = representatorFactories
        
    def getRepresentator(self, trainArticlesDB, articleAcronymDB, fold = "",executionTimeObserver= None):
        
        representators = [factory.getRepresentator(trainArticlesDB = trainArticlesDB,
                                                    articleAcronymDB = articleAcronymDB,
                                                    fold = fold, 
                                                    executionTimeObserver = executionTimeObserver) 
                          for factory in self.representatorFactories]
        
        types = [r.getType() for r in representators]
        return Representator_Concat(representators, representator_type='&'.join(types))


class Representator_Concat(TextRepresentator): 
    """
    take doc2vec vectors of labelled articles
    """

    def __init__(self, representators, representator_type=None):
        TextRepresentator.__init__(self, representator_type)
        self.representators = representators

    def _get_columns_number(self, listArraysX):
        columns_number = 0
        for featuresArray in listArraysX:
            firstArrayShape = featuresArray[0].shape
            if len(firstArrayShape) > 1:
                columns_number += firstArrayShape[1]
            else:
                columns_number += firstArrayShape[0]
        return columns_number

    def transform(self, X):
        #return [self.transformInstance(x) for x in X]
        #return [np.where(self.transformInstance(x) > 0, 1, 0) for x in X]
        
        listArraysX = [r.transform(X) for r in self.representators]
        row_number = len(listArraysX[0])
        columns_number = self._get_columns_number(listArraysX)
        
        csr_matrix((row_number,columns_number))
        
        concatX = []
        
        for i in range(0, len(X)):
            arraysInstance = [array[i] for array in listArraysX] 
            #concat = np.concatenate(arraysInstance)
            concat = hstack(arraysInstance)
            concatX.append(concat)
            
        finalConcat = vstack(concatX)

        return finalConcat
    