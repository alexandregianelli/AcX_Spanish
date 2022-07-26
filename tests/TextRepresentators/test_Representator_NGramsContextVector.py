'''
Created on Feb 2, 2020

@author: jpereira
'''
import unittest
from sklearn.feature_extraction import DictVectorizer
from collections import Counter

from nltk.tokenize import RegexpTokenizer
from TextRepresentators.Representator_New_Locality import Representator_New_Locality
import numpy as np
from numpy.testing import assert_array_almost_equal
from TextRepresentators.Representator_NGramsContextVector import Representator_NGramsContextVector
from TextRepresentators.Representator_NGramsContextVector import ngrams_split
from helper import TrainInstance, TestInstance, mergeDictsOfLists

class Representator_NGramsContextVector_Test(unittest.TestCase):
    sentenceDecay = 0.7
    paragraphDecay = 0.4

    """
        >>> mock = Mock()
    >>> cursor = mock.connection.cursor.return_value
    >>> cursor.execute.return_value = ['foo']
    >>> mock.connection.cursor().execute("SELECT 1")
    ['foo']
    """
    test_text1 = "I am a sentence TEST_ACRO. I am"
    test_text1_transform = {"I" : 0.4,
                            "am" : 0.4,
                            "a" : 0.2,
                            "sentence" : 0.2,
                            "I am": 0.6,
                            "am a" : 0.3,
                            "a sentence" : 0.3,
                            "sentence I": 0.3,
                            "I am a" : 0.5,
                            "am a sentence" : 0.5,
                            "a sentence I": 0.5,
                            "sentence I am": 0.5
                            }

        
        
    def test_ngrams_split(self):
        pass
    
    def test_transformInstance(self):
        test_id = 1
        
        ngrams = [1, 2, 3]
        weights = [0.2,0.3,0.5]
        articlesDB =  {test_id:self.test_text1.replace("TEST_ACRO", "TEST_EXPAN")}
        vocabulary = DictVectorizer()
        result_matrix = vocabulary.fit_transform(ngrams_split(ngrams,f) for f in articlesDB.values())
        maxC = 1.0
        
        rep = Representator_NGramsContextVector(ngrams, weights, articlesDB, vocabulary, maxC)
        trainInstance = TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN")
        resultTrain = rep.transform([trainInstance])
        
        test_text1_rep = vocabulary.transform(self.test_text1_transform).toarray()[0]
        assert_array_almost_equal(resultTrain[0], test_text1_rep)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()