from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.AcronymExpander import PredictiveExpander,FactoryPredictiveExpander

import logging
import math
import numpy as np
#import re
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from helper import TrainInstance

from cachetools import cached, LRUCache, keys
from text_preparation import get_expansion_without_spaces

logger = logging.getLogger(__name__)

# This is a kind of context vector but we give less weight to words far from the acronym
class Factory_Expander_Locality(FactoryPredictiveExpander):

    def __init__(self):
        FactoryPredictiveExpander.__init__(self, None)
                
    def getExpander(self, trainArticlesDB, acronymDB = None, articleAcronymDB = None, fold="", executionTimeObserver = None):
        
        return Expander_Locality(trainArticlesDB)


class Expander_Locality(PredictiveExpander):

    
    def __init__(self, articlesDB, sentenceDecay = 0.5, expander_type=AcronymExpanderEnum.locality):
        PredictiveExpander.__init__(self, expander_type, None)
        self.articlesDB = articlesDB
        self.cacheModelsPerAcronym = {}
        self.stop_words = set(stopwords.words('english'))
        self.p_stemmer = PorterStemmer()
        self.sent_tokenizer = sent_tokenize
        self.word_tokenizer = RegexpTokenizer(r'\w+').tokenize
        self.sentenceDecay = sentenceDecay


    def split_list(self, a_list):
        half = len(a_list)//2
        return a_list[:half], a_list[half:]

    def _addWeightToTokens(self, weight, tokens, weightedTokens):
        for token in tokens:
            preProcessedToken = token.lower() # TODO consider stem also
            
            # a list of weights for each word
            weightedTokens.setdefault(preProcessedToken, []).append(weight)

    def _weightTokensInSentences(self, sent_list, weightedTokens):
        for i, sent in enumerate(sent_list):
            tokens = self.word_tokenizer(sent)
            weight = math.pow(self.sentenceDecay, float(i+1))
            self._addWeightToTokens(weight, tokens, weightedTokens)

    def getWeightedTokens(self, text, concept, weightedTokens):
        conceptsSurround = text.split(concept)
        
        #First string as only an acronym after it
        sentences = self.sent_tokenizer(conceptsSurround[0])
        # We reserve because the function expects that the first sentences are closer to the concept
        sentences.reverse()
        self._weightTokensInSentences(sentences, weightedTokens)

        # These strings are in the middle of two concepts
        # we divide the sentence list in half so we choose the closest sentence to the concept
        for surroundText in conceptsSurround[1:-1]:
            #TODO tokenize by paragraph
            sentences = self.sent_tokenizer(surroundText)
            first_list, second_list = self.split_list(sentences)
            
            self._weightTokensInSentences(first_list, weightedTokens)
            second_list.reverse()
            self._weightTokensInSentences(second_list, weightedTokens)
            
        
        #Last string as only an acronym before
        sentences = self.sent_tokenizer(conceptsSurround[-1])
        self._weightTokensInSentences(sentences, weightedTokens)
        
        return weightedTokens

    # We have to use None as default, otherwise python will not generate a new dict, instead it uses the previous dict more info: https://docs.python-guide.org/writing/gotchas/
    def transformInstance(self, x, weightedTokens = None):
        if weightedTokens == None:
            weightedTokens = {}
            
        if isinstance(x, TrainInstance):
            concept = get_expansion_without_spaces(x.expansion)

        else:
            concept = x.acronym
        
        text = x.getText(self.articlesDB)
                
        return self.getWeightedTokens(text, concept, weightedTokens)
                
        


    def transform(self, X):
        expansionsDict = {}
        for x in X:
            if isinstance(x, TrainInstance):
                expansion = x.expansion
                if expansion not in expansionsDict:
                    expansionsDict[expansion] = []
                expansionsDict[expansion].append(x)
            else:
                return [self.transformInstance(x) for x in X]


        expansionsWeightedTokens = {}
        for expansion, listX in expansionsDict.items():
            expansionsWeightedTokens[expansion] = {}
            for x in listX:
                self.transformInstance(x, expansionsWeightedTokens[expansion])

        return [expansionsWeightedTokens[x.expansion] for x in X]

    def fit(self, X_train, y_train):
        # Does nothing
        self.X_train = X_train
        self.y_train = y_train


    def _computeTokenSim(self, wl1, wl2):
        sortl1 = sorted(wl1, reverse=True)
        sortl2 = sorted(wl2, reverse = True)
        
        if len(sortl1) > len(sortl2):
            sortl1 = sortl1[:len(sortl2)]
        else:
            sortl2 = sortl2[:len(sortl1)]
            
        sim = 0.0
        for w1, w2 in zip(sortl1, sortl2):
            sim += w1 * w2
        
        return sim
    
    def computeSim(self, x1, x2):
        sim = 0.0
        
        if len(x1) > len(x2):
            tmp = x1
            x1 = x2
            x2 = tmp
        
        for token, x1WeightList in x1.items():
            x2WeightList = x2.get(token)
            if x2WeightList:
                sim += self._computeTokenSim(x1WeightList, x2WeightList)
                
        return sim
    
    # TODO Cache
    # @cached(cache=LRUCache(maxsize=10), key=lambda _, acronym, X_train_vec: keys.hashkey(acronym))
    def selfSim(self, x):
        return self.computeSim(x,x)
        
    def normalizedSim (self, x1, x2):
        return self.computeSim(x1,x2) / math.sqrt(self.selfSim(x1), self.selfSim(x2))
        

    def getSimilarities(self, X1, X2, normalized = False):
        sim_matrix = np.zeros((len(X1),len(X2)))
        
        for i1, x1 in enumerate(X1):
            for i2, x2 in enumerate(X2):
                sim_matrix[i1][i2] = self.computeSim(x1,x2)
        
        return sim_matrix

    def predict(self, X_test, acronym):
        
        sim_matrix = self.getSimilarities(self.X_train, X_test)
        
        argmax_array = np.argmax(sim_matrix, axis=0)
        labels = [self.y_train[idx] for idx in argmax_array]
        confidences = sim_matrix[argmax_array,np.arange(len(argmax_array))]
        #confidences = sim_matrix.take(argmax_array)
        return labels, confidences

        