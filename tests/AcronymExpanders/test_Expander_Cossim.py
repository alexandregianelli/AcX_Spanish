'''
Created on Feb 9, 2019

@author: jpereira
'''
import unittest
from AcronymExpanders.Expander_Cossim import Expander_Cossim
import numpy as np
from TextRepresentators.TextRepresentator import TextRepresentator

class Expander_Cossim_Test(unittest.TestCase):

    """
        >>> mock = Mock()
    >>> cursor = mock.connection.cursor.return_value
    >>> cursor.execute.return_value = ['foo']
    >>> mock.connection.cursor().execute("SELECT 1")
    ['foo']
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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()