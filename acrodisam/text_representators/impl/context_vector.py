
import numpy as np
from scipy.sparse import csr_matrix, vstack

from text_representators._base import TextRepresentatorFactory, TextRepresentator
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from helper import TrainInstance,\
    ExecutionTimeObserver
from text_preparation import get_expansion_without_spaces

import text_preparation
from typing import Optional
from inputters import TrainDataManager
from run_config import RunConfig
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

class FactoryContextVector(TextRepresentatorFactory):

    #def __init__(self, datasetName=None, saveAndLoad=False):
    #    pass
    
    def __init__(self, normalize: bool = True, run_config: RunConfig = RunConfig()):
        self.normalize = normalize
        
    def get_text_representator(
        self,
        train_data_manager: TrainDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TextRepresentator:
        """
        vocabulary = DictVectorizer()
        
        # discover corpus and vectorize file word frequencies in a single pass
        if execution_time_observer:
            execution_time_observer.start()
            
        #vocabulary.fit_transform([Counter(word_tokenize(f)) for f in trainArticlesDB.values()])
        
        result_matrix = vocabulary.fit_transform(articlesAsCounters(articles_db))
        
        if self.normalize:
            # get the maximum number of occurrences of the same word in the corpus, this is for normalization purposes
            maxC = result_matrix.sum(axis=1).max()
        else:
            maxC = 1
        
        if execution_time_observer:
            execution_time_observer.stop()
        #return Representator_ContextVector(trainArticlesDB, vocabulary, float(1))
        return RepresentatorContextVector(articles_db, vocabulary, maxC)
        """
        return RepresentatorContextVector(train_data_manager.get_preprocessed_articles_db(), None, 1.0)

class RepresentatorContextVector(TextRepresentator):

    def __init__(self, articlesDB, vocabulary, maxC):
        #TextRepresentator.__init__(self)
        #self.vocabulary = vocabulary
        self.articlesDB = articlesDB
        self.maxC = maxC
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
        