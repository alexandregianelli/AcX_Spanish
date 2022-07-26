'''
Created on Feb 9, 2019

@author: jpereira
'''
import unittest
from sklearn.feature_extraction import DictVectorizer
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import RegexpTokenizer
from TextRepresentators.Representator_New_Locality_Small_Para import Representator_New_Locality_Small_Para, regex_word_tokenizer
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from TextRepresentators.TextRepresentator import TextRepresentator
from helper import TrainInstance, TestInstance, mergeDictsOfLists


class Expander_New_Locality_Small_Para_Test(unittest.TestCase):
    sentenceDecay = 0.7
    paragraphDecay = 0.4
    paragrahDistance = 1

    """
        >>> mock = Mock()
    >>> cursor = mock.connection.cursor.return_value
    >>> cursor.execute.return_value = ['foo']
    >>> mock.connection.cursor().execute("SELECT 1")
    ['foo']
    """
    test_decay = 0.1
    test_text1 = "I am a sentence TEST_ACRO. I am another sentence. And another, one i i. \nA paragraph"
    test_text1_transform = {"I" : 2,
                            "am" : 2,
                            "a" : 1,
                            "sentence" : 2,
                            "another": 1.4,
                            "And" : 0.7,
                            "one" : 0.7,
                            "i" : 1.4,
                            "A" : 0.4,
                            "paragraph" : 0.4
                            }
    
    test_text1_transform_weights = {
                            "am" : 1,
                            "sentence" : 1,
                            "another": 0.7,
                            "and" : 0.7,
                            "one" : 0.7,
                            "paragraph" : 0.4
                            }

    test_text1_reverse = "A paragraph.\n And another, one i i . I am another sentence. I am a sentence TEST_ACRO"

    test_text2 = "I am a sentence TEST_ACRO. \n A new paragraph. I am another TEST_ACRO sentence. And another, one i i \n Old paragraph"
    test_text2_transform = {"I" : 2,
                            "am" : 2,
                            "a" : 1,
                            "sentence" : 2,
                            "A" : 0.7,
                            "new" : 0.7,
                            "paragraph" : 1.4,
                            "another": 2,
                            "And" : 0.7,
                            "one" : 0.7,
                            "i" : 1.4,
                            "Old" : 0.4,
                            }

    test_text2_transform_weights = {
                            "am" : 1,
                            "sentence" : 1,
                            "new" : 0.7,
                            "paragraph" : 0.7,
                            "another": 1,
                            "and" : 0.7,
                            "one" : 0.7,
                            "old" : 0.4,
                            }

    test_texts_transform_context_vector = {
                            "I" : 4,
                            "am" : 4,
                            "a" : 2,
                            "sentence" : 4,
                            "another": 4,
                            "And" : 1.4,
                            "one" : 1.4,
                            "i" : 2.8,
                            "A" : 1.4,
                            "paragraph" : 2.1,
                            "new" : 0.7,
                            "Old" : 0.4
                            }


    test_text1_2 = "I am a sentence TEST_ACRO. I am another sentence. And another, one i i. \nA paragraph\n Another paragraph a I am"
    test_text2_2 = "I am a sentence TEST_ACRO. \n A new paragraph. I am another TEST_ACRO sentence. And another, one i i \n Old paragraph \n one more parahraph bla bla"

        
    def assertListAlmostEqual(self, l1, l2, msg=None, places=None):
        self.assertEqual(len(l1), len(l2))
        for i1, i2 in zip(sorted(l1), sorted(l2)):
            self.assertAlmostEqual(i1, i2, places, msg)
    
    def assertTransfDictAlmostEqual(self, d1, d2, msg=None, places=None):
        self.assertEqual(d1.keys(), d2.keys())
        
        for key, value in d1.items():
            self.assertListAlmostEqual(value, d2[key], places=places, msg=msg)
    
    def test_transform(self):
        word_tokenizer = RegexpTokenizer(r'\w+').tokenize

        test_id = 1
        # Train instance test
        articlesDB =  {test_id:self.test_text1.replace("TEST_ACRO", "TEST_EXPAN")}
        vocabulary = DictVectorizer()
        vocabulary.fit_transform(Counter(word_tokenizer(f)) for f in articlesDB.values())
        test_text1_rep = vocabulary.transform(self.test_text1_transform).toarray()[0]

        rep = Representator_New_Locality_Small_Para(articlesDB,
                                         word_tokenizer = word_tokenizer, 
                                         vocabulary = vocabulary, 
                                         vectorizer = None, 
                                         sentenceDecay = self.sentenceDecay,
                                          paragraphDecay = self.paragraphDecay,
                                          paragraphDistance=self.paragrahDistance)
        
        trainInstance = TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN")
        resultTrain = rep.transform([trainInstance])
        assert_array_almost_equal(resultTrain[0], test_text1_rep)
       
        
        # Test instance test
        testInstance = TestInstance(-1, self.test_text1, "TEST_ACRO")
        resultTest = rep.transform([testInstance])
        assert_array_almost_equal(resultTest[0], test_text1_rep)
        
        # Limit case, start with acronym
        newText = "TEST_ACRO " + self.test_text1.replace("TEST_ACRO", "")
        testInstance = TestInstance(-2, newText, "TEST_ACRO")
        resultTest = rep.transform([testInstance])
        assert_array_almost_equal(resultTest[0], test_text1_rep)
        
        # Limit case, end with acronym
        testInstance = TestInstance(-3, self.test_text1_reverse, "TEST_ACRO")
        resultTest = rep.transform([testInstance])
        assert_array_almost_equal(resultTest[0], test_text1_rep)
        
        
        # test with two acronyms
        testInstance = TestInstance(-4, self.test_text2, "TEST_ACRO")
        resultTest = rep.transform([testInstance])
        test_text2_rep = vocabulary.transform(self.test_text2_transform).toarray()[0]
        assert_array_almost_equal(resultTest[0], test_text2_rep)
        
        
        # Train with two instances one expansion
        test_id2 = 2
        trainInstanceArticles = {test_id:self.test_text1.replace("TEST_ACRO", "TEST_EXPAN"),
                           test_id2:self.test_text2.replace("TEST_ACRO", "TEST_EXPAN")}
        vocabulary = DictVectorizer()
        vocabulary.fit_transform(Counter(word_tokenizer(f)) for f in trainInstanceArticles.values())
        rep = Representator_New_Locality_Small_Para(trainInstanceArticles, 
                                         word_tokenizer = word_tokenizer, 
                                         vocabulary = vocabulary, 
                                         vectorizer = None, 
                                         sentenceDecay = self.sentenceDecay,
                                         paragraphDecay = self.paragraphDecay,
                                         paragraphDistance=self.paragrahDistance)
        
        trainInstances = [TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN"),
                           TrainInstance(test_id2, "TEST_ACRO", "TEST_EXPAN")]
        resultTrain = rep.transform(trainInstances)
        test_text1_rep = vocabulary.transform(self.test_text1_transform).toarray()[0]
        test_text2_rep = vocabulary.transform(self.test_text2_transform).toarray()[0]

        assert_array_almost_equal(resultTrain[0], test_text1_rep)
        assert_array_almost_equal(resultTrain[1], test_text2_rep)
        
    def test_transform_ContextVector(self):
        word_tokenizer = RegexpTokenizer(r'\w+').tokenize

        test_id = 1
        # Train instance test
        articlesDB =  {test_id:self.test_text1.replace("TEST_ACRO", "TEST_EXPAN")}
        vocabulary = DictVectorizer()
        vocabulary.fit_transform(Counter(word_tokenizer(f)) for f in articlesDB.values())
        test_text1_rep = vocabulary.transform(self.test_text1_transform).toarray()[0]

        rep = Representator_New_Locality_Small_Para(articlesDB,
                                         word_tokenizer = word_tokenizer, 
                                         vocabulary = vocabulary, 
                                         vectorizer = None, 
                                         sentenceDecay = self.sentenceDecay,
                                          paragraphDecay = self.paragraphDecay,
                                         paragraphDistance=self.paragrahDistance,
                                          contextVectorMode = True)
        
        trainInstance = TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN")
        resultTrain = rep.transform([trainInstance])
        assert_array_almost_equal(resultTrain[0], test_text1_rep)
       
        
        # Test instance test
        testInstance = TestInstance(-1, self.test_text1, "TEST_ACRO")
        resultTest = rep.transform([testInstance])
        assert_array_almost_equal(resultTest[0], test_text1_rep)
        
        
        
        # Train with two instances one expansion
        test_id2 = 2
        trainInstanceArticles = {test_id:self.test_text1.replace("TEST_ACRO", "TEST_EXPAN"),
                           test_id2:self.test_text2.replace("TEST_ACRO", "TEST_EXPAN")}
        vocabulary = DictVectorizer()
        vocabulary.fit_transform(Counter(word_tokenizer(f)) for f in trainInstanceArticles.values())
        rep = Representator_New_Locality_Small_Para(trainInstanceArticles, 
                                         word_tokenizer = word_tokenizer, 
                                         vocabulary = vocabulary, 
                                         vectorizer = None, 
                                         sentenceDecay = self.sentenceDecay,
                                         paragraphDecay = self.paragraphDecay,
                                         paragraphDistance=self.paragrahDistance,
                                         contextVectorMode = True)
        
        trainInstances = [TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN"),
                           TrainInstance(test_id2, "TEST_ACRO", "TEST_EXPAN")]
        resultTrain = rep.transform(trainInstances)
        
        
        assert_array_equal(resultTrain[0], resultTrain[1])
        
        test_textscontext_vector_rep = vocabulary.transform(self.test_texts_transform_context_vector).toarray()[0]

        assert_array_almost_equal(resultTrain[0], test_textscontext_vector_rep)


    def test_with_more_paragraphs(self):
        word_tokenizer = RegexpTokenizer(r'\w+').tokenize

        test_id = 1
        # Train instance test
        articlesDB =  {test_id:self.test_text1_2.replace("TEST_ACRO", "TEST_EXPAN")}
        vocabulary = DictVectorizer()
        vocabulary.fit_transform(Counter(word_tokenizer(f)) for f in articlesDB.values())
        test_text1_rep = vocabulary.transform(self.test_text1_transform).toarray()[0]

        rep = Representator_New_Locality_Small_Para(articlesDB,
                                         word_tokenizer = word_tokenizer, 
                                         vocabulary = vocabulary, 
                                         vectorizer = None, 
                                         sentenceDecay = self.sentenceDecay,
                                          paragraphDecay = self.paragraphDecay,
                                          paragraphDistance=self.paragrahDistance)
        
        trainInstance = TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN")
        resultTrain = rep.transform([trainInstance])
        assert_array_almost_equal(resultTrain[0], test_text1_rep)
        
        
        
        # Train with two instances one expansion
        test_id2 = 2
        trainInstanceArticles = {test_id:self.test_text1_2.replace("TEST_ACRO", "TEST_EXPAN"),
                           test_id2:self.test_text2_2.replace("TEST_ACRO", "TEST_EXPAN")}
        vocabulary = DictVectorizer()
        vocabulary.fit_transform(Counter(word_tokenizer(f)) for f in trainInstanceArticles.values())
        rep = Representator_New_Locality_Small_Para(trainInstanceArticles, 
                                         word_tokenizer = word_tokenizer, 
                                         vocabulary = vocabulary, 
                                         vectorizer = None, 
                                         sentenceDecay = self.sentenceDecay,
                                         paragraphDecay = self.paragraphDecay,
                                         paragraphDistance=self.paragrahDistance,
                                         contextVectorMode = True)
        
        trainInstances = [TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN"),
                           TrainInstance(test_id2, "TEST_ACRO", "TEST_EXPAN")]
        resultTrain = rep.transform(trainInstances)
        
        
        assert_array_equal(resultTrain[0], resultTrain[1])
        
        test_textscontext_vector_rep = vocabulary.transform(self.test_texts_transform_context_vector).toarray()[0]

        assert_array_almost_equal(resultTrain[0], test_textscontext_vector_rep)
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()