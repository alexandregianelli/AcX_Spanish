
import numpy as np
from scipy.sparse import csr_matrix, vstack

from TextRepresentators import TextRepresentatorEnum
from TextRepresentators.TextRepresentator import TextRepresentator, TextRepresentatorFactory
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from helper import TrainInstance
from text_preparation import get_expansion_without_spaces


import text_preparation
word_tokenize = text_preparation.tokenizePreProcessedArticle

from Logger import logging

logger = logging.getLogger(__name__)


def sparse_memory_usage(mat):
    try:
        return (mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes) / 1000000.0
    except AttributeError:
        return -1

def articlesAsCounters(trainArticlesDB):
    for f in trainArticlesDB.values():
        yield Counter(word_tokenize(f))

class Factory_ContextVector(TextRepresentatorFactory):

    #def __init__(self, datasetName=None, saveAndLoad=False):
    #    pass
    
    def __init__(self, normalize = True):
        self.normalize = normalize
        
    def getRepresentator(self, trainArticlesDB, articleAcronymDB, fold = "", executionTimeObserver= None):
        
        vocabulary = DictVectorizer()
        
        # discover corpus and vectorize file word frequencies in a single pass
        if executionTimeObserver:
            executionTimeObserver.start()
            
        #vocabulary.fit_transform([Counter(word_tokenize(f)) for f in trainArticlesDB.values()])
        """
        result_matrix = vocabulary.fit_transform(articlesAsCounters(trainArticlesDB))
        
        if self.normalize:
            # get the maximum number of occurrences of the same word in the corpus, this is for normalization purposes
            maxC = result_matrix.sum(axis=1).max()
        else:
            maxC = 1
        """
        if executionTimeObserver:
            executionTimeObserver.stop()
        return Representator_ContextVector(trainArticlesDB, vocabulary, float(1))


class Representator_ContextVector(TextRepresentator):

    def __init__(self, articlesDB, vocabulary, maxC, representator_type=TextRepresentatorEnum.Text_Embedding):
        TextRepresentator.__init__(self, representator_type)
        #self.vocabulary = vocabulary
        self.articlesDB = articlesDB
        #self.maxC = maxC
        #print("maxC: " + str(maxC))
        #self.vector_size = len(self.vocabulary.get_feature_names())

    def _divideSparseMatrix(self, m, s):
        m /= s
        return m

    def transform(self, X):
        
        item = X[0]
        if not isinstance(item, TrainInstance):
            return [self._divideSparseMatrix(self.transformInstance(x), self.maxC) for x in X]
        
        iter_counters = self.transform_train_instances(X)
        self.vocabulary = DictVectorizer(sparse=False)
        result_matrix = self.vocabulary.fit_transform(iter_counters)
        self.maxC = result_matrix.sum(axis=1).max()
        result_matrix = self._divideSparseMatrix(result_matrix, self.maxC)
        #result_matrix = result_matrix.toarray()
        self.vector_size = len(self.vocabulary.get_feature_names())

        
        expansionsEmbeddings = {}
        for i, x in enumerate(X):
            expansion = x.expansion
            if expansion not in expansionsEmbeddings:
                #expansionsEmbeddings[expansion] = csr_matrix((1,self.vector_size), dtype=np.float64)
                expansionsEmbeddings[expansion] = np.zeros((1,self.vector_size), dtype=np.float64)
            expansionsEmbeddings[expansion] += result_matrix[i]

        return [expansionsEmbeddings[x.expansion] for x in X]


    def transform_train_instances(self, X):
        for x in X:
            concept = get_expansion_without_spaces(x.expansion)

            text = x.getText(self.articlesDB).replace(concept,'')
        
            tokens = word_tokenize(text)
            counter = Counter(tokens)
            yield counter
            
    def transformInstance(self, x):            
        concept = x.acronym
        
        text = x.getText(self.articlesDB).replace(concept,'')
        
        tokens = word_tokenize(text)
        
        return self.vocabulary.transform(Counter(tokens))
        