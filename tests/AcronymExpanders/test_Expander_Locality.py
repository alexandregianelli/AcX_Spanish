'''
Created on Feb 9, 2019

@author: jpereira
'''
import unittest
from AcronymExpanders.Expander_Locality import Expander_Locality
import numpy as np
from numpy.testing import assert_array_almost_equal
from TextRepresentators.TextRepresentator import TextRepresentator
from helper import TrainInstance, TestInstance, mergeDictsOfLists

class Expander_Locality_Test(unittest.TestCase):


    """
        >>> mock = Mock()
    >>> cursor = mock.connection.cursor.return_value
    >>> cursor.execute.return_value = ['foo']
    >>> mock.connection.cursor().execute("SELECT 1")
    ['foo']
    """
    test_decay = 0.1
    test_text1 = "I am a sentence TEST_ACRO. I am another sentence. And another, one i i"
    test_text1_transform = {"i" : [0.001, 0.001, 0.01, 0.1],
                            "am" : [0.01, 0.1],
                            "a" : [0.1],
                            "sentence" : [0.01, 0.1],
                            "another": [0.001, 0.01],
                            "and" : [0.001],
                            "one" : [0.001]
                            }

    test_text1_reverse = "And another, one i i . I am another sentence. I am a sentence TEST_ACRO"

    test_text2 = "I am a sentence TEST_ACRO. I am another TEST_ACRO sentence. And another, one i i"
    test_text2_transform = {"i" : [0.01, 0.01, 0.1, 0.1],
                            "am" : [0.1, 0.1],
                            "a" : [0.1],
                            "sentence" : [0.1, 0.1],
                            "another": [0.01, 0.1],
                            "and" : [0.01],
                            "one" : [0.01]
                            }

    x1 = {
        "a" : [0.1,0.001],
        "b" : [0.4,0.3,0.2,0.1],
        "c" : [0.2,0.1,0.01,0.0004]            
        }
    
    x2 = {
        "a" : [0.0001,0.3,0.0001],
        "c" : [0.01],
        "d" : [0.2,0.1,0.4]
        }
    
    x3 = {
        "b" : [0.4, 0.01],
        "c" : [0.0001],
        "x" : [0.1,0.2,0.4]
        }
        
    def assertListAlmostEqual(self, l1, l2, msg=None, places=None):
        self.assertEqual(len(l1), len(l2))
        for i1, i2 in zip(sorted(l1), sorted(l2)):
            self.assertAlmostEqual(i1, i2, places, msg)
    
    def assertTransfDictAlmostEqual(self, d1, d2, msg=None, places=None):
        self.assertEqual(d1.keys(), d2.keys())
        
        for key, value in d1.items():
            self.assertListAlmostEqual(value, d2[key], places=places, msg=msg)
    
    
    def test_transform(self):
        test_id = 1
        # Train instance test
        exp = Expander_Locality({test_id:self.test_text1.replace("TEST_ACRO", "TEST_EXPAN")}, self.test_decay)
        trainInstance = TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN")
        resultTrain = exp.transform([trainInstance])
        self.assertTransfDictAlmostEqual(resultTrain[0], self.test_text1_transform)
        
        # Test instance test
        testInstance = TestInstance(-1, self.test_text1, "TEST_ACRO")
        resultTest = exp.transform([testInstance])
        self.assertTransfDictAlmostEqual(resultTest[0], self.test_text1_transform)
        
        # Limit case, start with acronym
        newText = "TEST_ACRO " + self.test_text1.replace("TEST_ACRO", "")
        testInstance = TestInstance(-2, newText, "TEST_ACRO")
        resultTest = exp.transform([testInstance])
        self.assertTransfDictAlmostEqual(resultTest[0], self.test_text1_transform)
        
        # Limit case, end with acronym
        testInstance = TestInstance(-3, self.test_text1_reverse, "TEST_ACRO")
        resultTest = exp.transform([testInstance])
        self.assertTransfDictAlmostEqual(resultTest[0], self.test_text1_transform)
        
        # test with two acronyms
        testInstance = TestInstance(-4, self.test_text2, "TEST_ACRO")
        resultTest = exp.transform([testInstance])
        self.assertTransfDictAlmostEqual(resultTest[0], self.test_text2_transform)
        
        # Train with two instances one expansion
        test_id2 = 2
        trainInstanceArticles = {test_id:self.test_text1.replace("TEST_ACRO", "TEST_EXPAN"),
                           test_id2:self.test_text2.replace("TEST_ACRO", "TEST_EXPAN")}
        exp = Expander_Locality(trainInstanceArticles, self.test_decay)
        
        trainInstances = [TrainInstance(test_id, "TEST_ACRO", "TEST_EXPAN"),
                           TrainInstance(test_id2, "TEST_ACRO", "TEST_EXPAN")]
        resultTrain = exp.transform(trainInstances)
        correctResult = mergeDictsOfLists(self.test_text1_transform, self.test_text2_transform)
        self.assertTransfDictAlmostEqual(resultTrain[0], correctResult)
   
    #def test_fit(self):
        
        #todo
        
    def test_computeSim(self):
        
        exp = Expander_Locality({}, self.test_decay)
        result = exp.computeSim(self.x1,self.x2)
        self.assertAlmostEqual(result, 0.0320001)
        
        # Self
        result = exp.computeSim(self.x1,self.x1)
        self.assertAlmostEqual(result, 0.36010116)
        
        # Limit case  empty
        result = exp.computeSim(self.x1,{})
        self.assertEqual(result, 0.0)
        
        # Limit case, expect 0
        result = exp.computeSim(self.x1,{"z": [0.1,0.2], "x": [0.001]})
        self.assertEqual(result, 0.0)
        
        
    def test_selfSim(self):
        exp = Expander_Locality({}, self.test_decay)
        result = exp.selfSim(self.x1)
        self.assertAlmostEqual(result, 0.36010116)
        
        result = exp.selfSim({})
        self.assertAlmostEqual(result, 0.0)
        
    #def test_normalizedSim(self):
        #TODO
        
    def test_getSimilarities(self):
        
        exp = Expander_Locality({}, self.test_decay)
        result_sim_matrix = exp.getSimilarities([self.x1, {}],[self.x2, self.x3])
        sim_matrix = np.matrix([[0.0320001, 0.16302],[0.0, 0.0]])
        
        assert_array_almost_equal(result_sim_matrix, sim_matrix)
        
        
    def test_predict(self):
        l2 = "exp2"
        l3 = "exp3"
        exp = Expander_Locality({}, self.test_decay)
        X_train = [self.x2, self.x3]
        y_train = [l2, l3]
        exp.fit(X_train, y_train)
        labels, confidences = exp.predict([self.x1, {"a":[0.1]}, {"c":[0.1]}], None)
        self.assertEqual(labels, [l3,l2,l2])
        
        for pConf, tConf in zip(confidences, [0.16302, 0.03, 0.001]):
            self.assertAlmostEqual(pConf,tConf)

    """
    def test_fit(self):
        textRepresentator = None
        exp = Expander_Cossim(textRepresentator)
        X_train = [np.zeros(3),np.ones(3)]
        y_train = [0, 1]
        exp.fit(X_train, y_train)
        self.assertEqual(exp.X_train, X_train)
        self.assertEqual(exp.y_train, y_train)


    def test_predict(self):
        textRepresentator = None
        exp = Expander_Cossim(textRepresentator)
        X_train = [np.zeros(4),np.ones(4), np.array([0,4,0,4])]
        y_train = [0, 1, 2]
        exp.fit(X_train, y_train)
        acronym = None
        
        labels, cofidence = exp.predict([np.zeros(4)], acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[0], labels[0])
        self.assertEqual(0, cofidence[0])
        
        
        labels, cofidence = exp.predict([np.ones(4)], acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[1], labels[0])
        self.assertEqual(1, cofidence[0])
        
        labels, cofidence = exp.predict([np.array([1,1,2,2])],acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[1], labels[0])
        self.assertAlmostEqual(cofidence[0], 0.9487, places=4)
        
        labels, cofidence = exp.predict([np.divide(np.array([1,1,2,2]), 20)],acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[1], labels[0])
        self.assertAlmostEqual(cofidence[0], 0.9487, places=4)
        
        
        labels, cofidence = exp.predict([np.array([0,8,0,8])],acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[2], labels[0])
        self.assertAlmostEqual(1, cofidence[0])
        
        
    def test_predict_divide(self):
        divScalar = 341334
        textRepresentator = None
        exp = Expander_Cossim(textRepresentator)
        X_train = [np.zeros(4),np.divide(np.ones(4),divScalar), np.divide(np.array([0,4,0,4]),divScalar)]
        y_train = [0, 1, 2]
        exp.fit(X_train, y_train)
        acronym = None
        
        labels, cofidence = exp.predict([np.zeros(4)], acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[0], labels[0])
        self.assertEqual(0, cofidence[0])
        
        
        labels, cofidence = exp.predict([np.ones(4)], acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[1], labels[0])
        self.assertEqual(1, cofidence[0])
        
        labels, cofidence = exp.predict([np.divide(np.array([1,1,2,2]), divScalar)],acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[1], labels[0])
        self.assertAlmostEqual(cofidence[0], 0.9487, places=4)
        
        labels, cofidence = exp.predict([np.divide(np.divide(np.array([1,1,2,2]), divScalar), 20)],acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[1], labels[0])
        self.assertAlmostEqual(cofidence[0], 0.9487, places=4)
        
        
        labels, cofidence = exp.predict([np.divide(np.array([0,8,0,8]), divScalar)],acronym)
        self.assertEqual(len(labels), 1)
        self.assertEqual(y_train[2], labels[0])
        self.assertAlmostEqual(1, cofidence[0])
    """

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()