#Replaced by default
#from nltk.tokenize import word_tokenize
import numpy as np
from scipy.sparse import csr_matrix, vstack

from TextRepresentators import TextRepresentatorEnum
from TextRepresentators.TextRepresentator import TextRepresentator, TextRepresentatorFactory
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from helper import TrainInstance, get_expansion_without_spaces

import text_preparation
word_tokenize = text_preparation.tokenizePreProcessedArticle

def articlesAsCounters(trainArticlesDB):
    for f in trainArticlesDB.values():
        yield Counter(word_tokenize(f))

class Factory_ContextVector(TextRepresentatorFactory):

    #def __init__(self, datasetName=None, saveAndLoad=False):
    #    pass
        
    def getRepresentator(self, trainArticlesDB, articleAcronymDB, fold = "",executionTimeObserver= None):
        
        vocabulary = DictVectorizer()
        
        # discover corpus and vectorize file word frequencies in a single pass
        if executionTimeObserver:
            executionTimeObserver.start()
            
        #vocabulary.fit_transform([Counter(word_tokenize(f)) for f in trainArticlesDB.values()])
        
        result_matrix = vocabulary.fit_transform(articlesAsCounters(trainArticlesDB))
        
        # get the maximum number of occurrences of the same word in the corpus, this is for normalization purposes
        maxC = result_matrix.sum(axis=1).max()
        
        if executionTimeObserver:
            executionTimeObserver.stop()
        return Representator_ContextVector(trainArticlesDB, vocabulary, float(maxC))


class Representator_ContextVector(TextRepresentator):
    """
    take doc2vec vectors of labelled articles
    """

    def __init__(self, articlesDB, vocabulary, maxC, representator_type=TextRepresentatorEnum.Text_Embedding):
        TextRepresentator.__init__(self, representator_type)
        self.vocabulary = vocabulary
        self.articlesDB = articlesDB
        self.maxC = maxC
        #print("maxC: " + str(maxC))
        self.vector_size = len(self.vocabulary.get_feature_names())

    def _divideSparseMatrix(self, m, s):
        m /= s
        return m

    def transform(self, X):
        #return [self.transformInstance(x) for x in X]
        #return [np.where(self.transformInstance(x) > 0, 1, 0) for x in X]
        
        expansionsDict = {}
        for x in X:
            if isinstance(x, TrainInstance):
                expansion = x.expansion
                if expansion not in expansionsDict:
                    expansionsDict[expansion] = []
                expansionsDict[expansion].append(x)
            else:
                #return [np.true_divide(self.transformInstance(x),self.maxC) for x in X]
                return vstack([self._divideSparseMatrix(self.transformInstance(x), self.maxC) for x in X])
                #return [np.divide(self.transformInstance(x), self.maxC) for x in X]
                #return [np.where(self.transformInstance(x) > 0, 1, 0) for x in X]
        
        #vector_size = len(self.vocabulary.get_feature_names())
        #vector_size = 200

        expansionsEmbeddings = {}
        for expansion, listX in expansionsDict.items():
            #expansionsEmbeddings[expansion] = np.zeros(vector_size)
            expansionsEmbeddings[expansion] = csr_matrix((1,self.vector_size), dtype=np.float64)
            #expansionsEmbeddings[expansion] = array([0])#np.zeros(vector_size)
            for x in listX:
                expansionsEmbeddings[expansion] = expansionsEmbeddings[expansion] + self.transformInstance(x)
                
            expansionsEmbeddings[expansion] /= self.maxC

        #return [np.true_divide(expansionsEmbeddings[x.expansion], self.maxC) for x in X]
        return vstack([expansionsEmbeddings[x.expansion] for x in X])
        #return [np.divide(expansionsEmbeddings[x.expansion],self.maxC) for x in X]

    def transformInstance(self, x):            
        if isinstance(x, TrainInstance):
            concept = get_expansion_without_spaces(x.expansion)

        else:
            concept = x.acronym
        
        text = x.getText(self.articlesDB).replace(concept,'')
        
        tokens = word_tokenize(text)
        
        return self.vocabulary.transform(Counter(tokens))
        
        #return self.vocabulary.transform(Counter(tokens)).toarray()[0]
