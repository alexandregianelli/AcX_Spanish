#import time
import sys
#import gc
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, bmat

#from TextRepresentators_backup import TextRepresentatorEnum
from TextRepresentators.TextRepresentator import TextRepresentator, TextRepresentatorFactory
#from helper import TrainInstance, getExpansionWithoutSpaces

from Logger import logging

logger = logging.getLogger(__name__)

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

def sparse_memory_usage(mat):
    try:
        return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes / 1000000.0
    except AttributeError:
        return -1
    
    
class Representator_Concat(TextRepresentator): 

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

    def get_individual_transforms(self, X):
        listArraysX = [r.transform(X) for r in self.representators]
        for i in range(0, len(X)):
            #logger.debug("Processing ith X: " + str(i))
            yield [arr[i] for arr in listArraysX] 

    def transform(self, X):
        
        blocks = list(self.get_individual_transforms(X))
        #logger.debug("blocks: " + str(len(blocks)) + " " + str(sys.getsizeof(blocks)))
        #logger.debug("bmat")
        #gc.collect()
        #time.sleep(60)
        #finalConcat = blocks
        #finalConcat = bmat(blocks, format='csr')
        #finalConcat = bmat(blocks, format='coo')
        finalConcat = np.block(blocks)
        
        logger.debug("finalConcat: " + str(finalConcat.shape) + " " \
                     #+ str(finalConcat.getnnz()) + " "\
                     + str(finalConcat.nbytes) + " "\
                     #+ finalConcat.count_nonzero() + " "\
                      + str(sys.getsizeof(finalConcat)))# + " " + str(sparse_memory_usage(finalConcat)) + "M")

        return finalConcat
    