'''
Created on Aug 31, 2018

@author: jpereira
'''

class TextRepresentatorFactory:
        
    def getRepresentator(self, trainArticlesDB, articleAcronymDB, fold, executionTimeObserver= None):
        pass

class TextRepresentator:

    def __init__(self, representator_type):
        """
        inputs:
        representator_type: the type of TextRepresentator that this representator implements
        """
        self.type = representator_type

    def transform(self, X):
        """
        transforms input list to form accepted by fit and predict function
        inputs:
        X (list): of helper.ExpansionChoice
        returns:
        result (list): of inputs to predict and fit functions
        """
        pass

    def getType(self):
        """
        returns: the AcronymExpanderEnum that this expander implements
        """
        return self.type