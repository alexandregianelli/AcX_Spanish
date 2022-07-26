'''

@author: jpereira
'''
import random
from Logger import logging

import numpy as np

from helper import getAcronymChoices, groupby_indexes_unsorted, zip_with_scalar

from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.AcronymExpander import PredictiveExpander, FactoryPredictiveExpander
import itertools

logger = logging.getLogger(__name__)


class Factory_Expander_Generic_LearnSim(FactoryPredictiveExpander):

    def __init__(self, maxSeqLen=160, n_epoch = 1, datasetName=None, saveAndLoad=False, persistentArticles=None):
        FactoryPredictiveExpander.__init__(self)
        random.seed(1337)
        
        try:
            self.maxSeqLen = int(maxSeqLen)
        except ValueError:
            self.maxSeqLen = None
            logger.warning("Set maxSeqLen to None, original value cannot be converted to int: " + maxSeqLen)
            
            
        try:
            self.n_epoch = int(n_epoch)
        except ValueError:
            self.n_epoch = 1
            logger.warning("Set n_epoch to 1, original value cannot be converted to int: " + n_epoch)
        
        self.datasetName = datasetName
        self.saveAndLoad = saveAndLoad
        
        self.persistentArticles = persistentArticles

    def preProcessInstance(self, acroInstance, trainArticlesDB, firstSentence= True, numSeq = None):
                        
        pass
        
    def getDataPairs(self, y_train):
        positivePairs = set()
        
        for y, indexList in groupby_indexes_unsorted(y_train):
                for t in itertools.combinations(indexList, 2):
                    positivePairs.add(t)
                                
        allPairs = set(itertools.combinations(range(len(y_train)), 2))
        negativePairs = allPairs - positivePairs
            
        if not len(positivePairs) == len(negativePairs):
            if len(positivePairs) < len(negativePairs):
                negativePairs = random.sample(negativePairs, k=len(positivePairs))
            else:
                positivePairs = random.sample(positivePairs, k=len(negativePairs))
                    
        return positivePairs, negativePairs
    
    def getAcronymTrainData(self, trainArticlesDB, X_train, positivePairs, negativePairs):
        leftTrainData = []
        rightTrainData = []
        trainLabels = []
        
        preProcessed = dict()
        for pair, label in itertools.chain(zip_with_scalar(positivePairs,1), zip_with_scalar(negativePairs,0)):
            trainInstance1 = preProcessed.get(pair[0])
            if trainInstance1 is None:
                trainInstance1 = self.preProcessInstance(X_train[pair[0]], trainArticlesDB, 
                                                    firstSentence = True,
                                                     numSeq = self.maxSeqLen)
                preProcessed[pair[0]] = trainInstance1

            trainInstance2 = preProcessed.get(pair[1])
            if trainInstance2 is None:
                trainInstance2 = self.preProcessInstance(X_train[pair[1]], trainArticlesDB, 
                                                    firstSentence = False,
                                                     numSeq = self.maxSeqLen)
                preProcessed[pair[1]] = trainInstance2
                
            leftTrainData.append(trainInstance1)
            rightTrainData.append(trainInstance2)
            trainLabels.append(label)
            
        return leftTrainData, rightTrainData, trainLabels
        
    def getTrainData(self, trainArticlesDB, acronymDB, datasetName = "", fold = "", persistentArticles = None):
        #TODO load if exits use dataset and fold
        #train_c, train_r, train_l = pickle.load(open("/home/jpereira/tmp/dataset-dualencoder-lstm/dataset/" + 'train.pkl', 'rb'))        

        
        leftTrainData = []
        rightTrainData = []
        trainLabels = []
        # Build train data, an acronym at a time
        for acronym in acronymDB.keys():
            X_train, y_train, labelToExpansion = getAcronymChoices(acronym, acronymDB)
            
            positivePairs, negativePairs = self.getDataPairs(y_train)
            acroLeftTrainData, acroRightTrainData, acroTrainLabels = self.getAcronymTrainData(trainArticlesDB, X_train, positivePairs, negativePairs)
            
            leftTrainData.extend(acroLeftTrainData)
            rightTrainData.extend(acroRightTrainData)
            trainLabels.extend(acroTrainLabels)
            
    
        leftTrainData = np.asarray(leftTrainData)
        rightTrainData = np.asarray(rightTrainData)
        trainLabels = np.asarray(trainLabels)
        
        # shuffle training set
        indices = np.arange(leftTrainData.shape[0])
        
        np.random.shuffle(indices)
    
    
        leftTrainData = leftTrainData[indices]
        rightTrainData = rightTrainData[indices]
        trainLabels = trainLabels[indices]
        
        if not self.maxSeqLen or self.maxSeqLen < 1:
            maxSeqLen = max([len(np.concatenate([leftText, rightText])) for leftText, rightText in zip(leftTrainData, rightTrainData)])
            logger.info("No MaxSeqLen, set to maximum: " + str(maxSeqLen))
        else:
            maxSeqLen = self.maxSeqLen
        
        return leftTrainData, rightTrainData, trainLabels, maxSeqLen
        
        
        
    def getExpander(self, trainArticlesDB, acronymDB = None, articleAcronymDB = None, fold="", executionTimeObserver = None):
         
 
        
        leftTrainData, rightTrainData, trainLabels, maxSeqLen = self.getTrainData(trainArticlesDB, 
                                                                                  acronymDB,
                                                                                  datasetName = self.datasetName,
                                                                                  fold = fold, 
                                                                                  persistentArticles = self.persistentArticles)

        
        model, preProcessRunningInstance = self.getModel(leftTrainData, rightTrainData, trainLabels, maxSeqLen, fold,  
                                          executionTimeObserver=executionTimeObserver)
        
        return Expander_Generic_LearnSim(model, trainArticlesDB, maxSeqLen, preProcessRunningInstance)


    # return an object that can be called for prediction x_train list/array x_test and a preProcessRunningInstance function
    def getModel(self, leftTrainData, rightTrainData, trainLabels, maxSeqLen, fold, executionTimeObserver=None):
        pass

class Expander_Generic_LearnSim(PredictiveExpander):

    def __init__(self, model, articlesDB, maxSeqLen, preProcessInstance, expander_type=AcronymExpanderEnum.none):
        PredictiveExpander.__init__(self, expander_type, None)
        self.model = model
        self.articlesDB = articlesDB
        self.preProcessInstance = preProcessInstance
        self.maxSeqLen = maxSeqLen
        
    def transform(self, X):
        X = super().transform(X)
        preProcessedX = [self.preProcessInstance(x, self.articlesDB, firstSentence = None, numSeq=self.maxSeqLen) for x in X]
        return preProcessedX
    
    def fit(self, X_train, y_train):
        
        self.X_train = X_train
        self.y_train = y_train        
        
        
    def predict(self, X_test, acronym):
        
        labels = []
        confidences = []
        
        for x_test in X_test:            
            # DO repeat
            probablities = self.model(self.X_train, x_test)   
        
            mostSimilarX =  np.argmax(probablities, axis=0)
            labels.append(self.y_train[mostSimilarX])
            confidences.append(probablities[mostSimilarX])
        
        
        return labels,  confidences
