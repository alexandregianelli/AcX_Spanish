'''

@author: jpereira
'''
import random
import os

from Logger import logging

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.utils import to_categorical

#import numpy as np
#import pickle
#from keras.models import Sequential
#from keras.utils import np_utils
#from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Activation
#from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Multiply

#import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import pickle

#import params_flow as pf

#import bert

from tensorflow.keras.layers import Input, Multiply, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from keras.layers import concatenate
from tensorflow.keras.models import load_model


import sentencepiece as spm

#import tensorflow as tf

#import argparse
#import numpy as np

import text_preparation
from helper import getAcronymChoices, groupby_indexes_unsorted, get_expansion_without_spaces, zip_with_scalar, TrainInstance, getDatasetGeneratedFilesPath

from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.AcronymExpander import PredictiveExpander, FactoryPredictiveExpander
import itertools

import bert

logger = logging.getLogger(__name__)

def preProcessInstance(acroInstance, trainArticlesDB, sentenceProcessor, firstSentence, numSeq = None):
                
    if firstSentence:
        numSeqArg = numSeq//2 - 2
    else:
        numSeqArg = numSeq//2 - 1

    tokens = text_preparation.preProcessArticle(acroInstance, 
                                         trainArticlesDB, 
                                         numSeq = numSeqArg, 
                                         tokenizer = lambda text: sentenceProcessor.encode_as_ids(text))
    
    if firstSentence:
        new_tokens = [sentenceProcessor.piece_to_id("[CLS]")] + tokens + [sentenceProcessor.piece_to_id("[SEP]")]
    else:
        new_tokens = tokens + [sentenceProcessor.piece_to_id("[SEP]")]


    spanId = sentenceProcessor.piece_to_id("<span>")
    seq = pad_sequences([new_tokens], maxlen=numSeq//2 , padding='post',truncating='post', value = spanId)
    
    if len(seq[0]) < len(tokens):
        logger.error("Pre processing failed, len(tokens) is higher than numseq")
        tokens = text_preparation.preProcessArticle(acroInstance, 
                                         trainArticlesDB, 
                                         numSeq = numSeqArg, 
                                         tokenizer = lambda text: sentenceProcessor.encode_as_ids(text))
    
    return seq[0]
    

class Factory_Expander_ALBERT(FactoryPredictiveExpander):

    def __init__(self, maxSeqLen=160, datasetName=None, saveAndLoad=False, persistentArticles=None):
        FactoryPredictiveExpander.__init__(self)
        random.seed(1337)
        
        try:
            self.maxSeqLen = int(maxSeqLen)
        except ValueError:
            self.maxSeqLen = None
            logger.info("Set maxSeqLen to None, original value cannot be converted to int: " + maxSeqLen)
        
        # TODO
        
        self.embeddings = 300 
        self.hidden_size = 300
        self.n_epochs = 1
        
        self.datasetName = datasetName
        self.saveAndLoad = saveAndLoad
        
        self.persistentArticles = persistentArticles
        
        #self.loss = args[0]
        #self.c = float(args[1])
                
        #train_c, train_r, train_l = pickle.load(open("/home/jpereira/tmp/dataset-dualencoder-lstm/dataset/" + 'train.pkl', 'rb'))
        #print("ddd")
        
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
            if not trainInstance1:
                #def preProcessInstance(acroInstance, trainArticlesDB, sentenceProcessor, firstSentence, numSeq = None):

                trainInstance1 = preProcessInstance(X_train[pair[0]], trainArticlesDB, 
                                                    sentenceProcessor = self.sp,
                                                    firstSentence = True,
                                                     numSeq = self.maxSeqLen)

            trainInstance2 = preProcessed.get(pair[1])
            if not trainInstance2:
                trainInstance2 = preProcessInstance(X_train[pair[1]], trainArticlesDB, 
                                                    sentenceProcessor = self.sp,
                                                    firstSentence = False,
                                                     numSeq = self.maxSeqLen)
            leftTrainData.append(trainInstance1)
            rightTrainData.append(trainInstance2)
            trainLabels.append(label)
            
        return leftTrainData, rightTrainData, trainLabels
        
    def getTrainData(self, trainArticlesDB, acronymDB, datasetName = "", fold = "", persistentArticles = None):
        #TODO load if exits use dataset and fold
        
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
            
            
        # Transform Data
        
        #tokenizer = Tokenizer(filters='')
        #tokenizer.fit_on_texts(leftTrainData)
        #tokenizer.fit_on_texts(rightTrainData)

        #train_c = [self.sp.encode_as_ids(' '.join(["[CLS]"] + text + ["[SEP]"])) for text in leftTrainData]
        #train_r = [self.sp.encode_as_ids(' '.join(text + ["[SEP]"])) for text in rightTrainData]
        
        train_c = leftTrainData
        train_r = rightTrainData

        train_ids = [np.concatenate([leftText, rightText]) for leftText, rightText in zip(train_c, train_r)]
        train_seq = [[0] * len(leftText) + [1] * len(rightText) for leftText, rightText in zip(train_c, train_r)]

        train_ids = np.array(train_ids,  dtype=np.int32)
        train_seq = np.array(train_seq,  dtype=np.int32)

        #train_c = tokenizer.texts_to_sequences(leftTrainData)
        #train_r = tokenizer.texts_to_sequences(rightTrainData)
        
        #train_c = leftTrainData
        #train_r = rightTrainData
    
        if not self.maxSeqLen or self.maxSeqLen < 1:
            maxSeqLen = max([len(seq) for seq in train_ids])
            logger.info("No MaxSeqLen, set to maximum: " + str(maxSeqLen))
        else:
            maxSeqLen = self.maxSeqLen
             
        #MAX_SEQUENCE_LENGTH = max([len(seq) for seq in train_c + train_r
                                                        #+ test_c + test_r
                                                        #+ dev_c + dev_r])
        
        #max_number_words = len(tokenizer.word_index) + 1
        #word_index = tokenizer.word_index
        
        
        #logger.info("MAX_NB_WORDS: {}".format(max_number_words))
        
        #train_ids = pad_sequences(train_ids, maxlen=maxSeqLen, padding='post',truncating='post', value = spanId)
        #train_seq = pad_sequences(train_seq, maxlen=maxSeqLen, padding='post',truncating='post', value=1)

        # shuffle training set
        indices = np.arange(train_ids.shape[0])
        
        np.random.shuffle(indices)
        train_l = trainLabels
    
        train_ids = np.asarray(train_ids)
        train_seq = np.asarray(train_seq)
        train_l = np.asarray(train_l)
    
        train_ids = train_ids[indices]
        train_seq = train_seq[indices]
        train_l = train_l[indices]        
        
        return train_ids, train_seq, train_l, maxSeqLen
        #TODO save into disk
        #pickle.dump([train_c, train_r, train_l], open(DATA_DIR + "train.pkl", "wb"), protocol=-1)
        #pickle.dump([test_c, test_r, test_l], open(DATA_DIR + "test.pkl", "wb"), protocol=-1)
        #pickle.dump([dev_c, dev_r, dev_l], open(DATA_DIR + "dev.pkl", "wb"), protocol=-1)

        #pickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open(DATA_DIR + "params.pkl", "wb"), protocol=-1)
    
    def getModel(self, train_ids, train_seq, train_l, maxSeqLen, embeddings="", hidden_size = 300, n_epochs = 1, executionTimeObserver=None):
        
        logger.info("Starting...")
        
        model_name = "albert_base"
        model_dir    = bert.fetch_tfhub_albert_model(model_name, "/home/jpereira/git/AcroDisam/acrodisam_app/data/albert/")
        model_params = bert.albert_params(model_name)
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

        # use in Keras Model here, and call model.build()

        
        #emb_dim = 300
        #hidden_size = 300
        optimizer = "adam"
        batch_size = 32 # for seq of 20
        #n_epochs = 50
        #n_epochs = 1
#        with tf.device('/cpu:0'):

        #pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
        #model3  = Model(inputs=model.input, outputs=pool_output)
        #model3.compile(loss='binary_crossentropy', optimizer=adam)
        #model3.summary()
        
        
        context_input_ids      = Input(shape=(maxSeqLen,), dtype='int32', name="context_input_ids")
        context_token_type_ids = Input(shape=(maxSeqLen,), dtype='int32', name="context_token_type_ids")
        context_output         = l_bert([context_input_ids, context_token_type_ids])
        
        # TODO ver bert QA
        context_output = keras.layers.Lambda(lambda x: x[:, 0, :])(context_output)
        


#         print("bert shape", output.shape)
#         output = keras.layers.Lambda(lambda x: x[:, 0, :])(output)
#         output = keras.layers.Dense(1, activation="sigmoid")(output)
#     
#         model = keras.Model(inputs=[input_ids, token_type_ids], outputs=output)
#         model.build(input_shape=[(None, max_seq_len)])
    
        
        logger.info("Now creating the model...")
        #with tf.device('/cpu:0'):
        # define lstm encoder
    
        
        #concatenated = concatenate([context_branch, response_branch], mode='mul')
        out = Dense((1), activation = "sigmoid") (context_output)
    
        #model = Model([context_input_ids, context_token_type_ids, response_input_ids, response_token_type_ids], out)
        model = Model([context_input_ids, context_token_type_ids], out)
        
        model.build(input_shape=[(None, maxSeqLen), (None, maxSeqLen)])
        
        model.compile(loss='binary_crossentropy',
                        optimizer=keras.optimizers.Adam())
        
        #logger.info(l_bert.summary())
        logger.info(model.summary()) 
        
        bert.load_albert_weights(l_bert, model_dir)      # should be called after model.build()
        
        #bert.load_albert_weights(l_bert, albert_dir)
    
        model.summary()
        
        logger.info('Found %s training samples.' % len(train_ids))
        
        #histories = my_callbacks.Histories()
        
        #bestAcc = 0.0
        #patience = 0 
        
        
        logger.info("\tbatch_size={}, nb_epoch={}".format(batch_size, n_epochs))
        if executionTimeObserver:
            executionTimeObserver.start()
        model.fit((train_ids, train_seq), train_l,
                    batch_size=batch_size, epochs=n_epochs, #callbacks=[histories],
                    #validation_data=([dev_c, dev_r], dev_l),
                     verbose=1)
        if executionTimeObserver:
            executionTimeObserver.stop()
        """
        for ep in range(1, n_epochs):
                    
            model.fit([train_c, train_r], train_l,
                    batch_size=batch_size, epochs=1, #callbacks=[histories],
                    #validation_data=([dev_c, dev_r], dev_l),
                     verbose=1)
        """
        #if save_model:
        #    print("Now saving the model... at {}".format(args.model_fname))
        #    model.save(args.model_fname)
        
        return model
    
            
        
    
    

        
    def getExpander(self, trainArticlesDB, acronymDB = None, articleAcronymDB = None, fold="", executionTimeObserver = None):
        
        self.sp_model = tf.io.gfile.glob(os.path.join("/home/jpereira/git/AcroDisam/acrodisam_app/data/albert/albert_base", "assets/*"))[0]
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.sp_model)
        
        
        if self.saveAndLoad:
            generatedFilesFolder = getDatasetGeneratedFilesPath(self.datasetName)
            varToName = "_".join([str(s) for s in [self.datasetName,fold, self.maxSeqLen, self.embeddings, self.hidden_size, self.n_epochs]])
            expander_albert_location = generatedFilesFolder + "EXPANDER_ALBERT" + "_" + varToName + ".pickle"
                
            if os.path.isfile(expander_albert_location):
                # TODO Load Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
                model, maxSeqLen = pickle.load(expander_albert_location)
                Expander_ALBERT(model, trainArticlesDB, self.sp , maxSeqLen) 
 
        
        train_ids, train_seq, train_l, maxSeqLen = self.getTrainData(trainArticlesDB, 
                                                                                                   acronymDB,
                                                                                                   datasetName = self.datasetName, 
                                                                                                   fold = fold, 
                                                                                                   persistentArticles = self.persistentArticles)
        #textRepresentator = self.getRepresentator(trainArticlesDB = trainArticlesDB,
        #                                          articleAcronymDB = articleAcronymDB,
        #                                          fold = fold,
        #                                          executionTimeObserver=executionTimeObserver)
        
        model = self.getModel(train_ids, train_seq, train_l, maxSeqLen, 
                                          embeddings=self.embeddings, hidden_size = self.hidden_size, n_epochs = self.n_epochs, 
                                          executionTimeObserver=executionTimeObserver)
        
        if self.saveAndLoad:
            # TODO save Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
            #dualEconder.save(dualEncoderLocation)
            pickle.dump([model, maxSeqLen], open(expander_albert_location, "wb"), protocol=-1)

        
        return Expander_ALBERT(model, trainArticlesDB, self.sp , maxSeqLen)


class Expander_ALBERT(PredictiveExpander):

    def __init__(self, dualEconder, articlesDB, sp, maxSeqLen, expander_type=AcronymExpanderEnum.none):
        PredictiveExpander.__init__(self, expander_type, None)
        self.dualEncoder = dualEconder
        self.articlesDB = articlesDB
        #self.tokenizer = tokenizer
        self.maxSeqLen = maxSeqLen
        self.sp = sp
    def transform(self, X):
        X = super().transform(X)

        return  X
    
    
    def bertInputGenerator(self, x_test):
        testIDs = preProcessInstance(x_test, self.articlesDB, 
                                            sentenceProcessor = self.sp,
                                            firstSentence = False,
                                            numSeq = self.maxSeqLen)
        
        #text = preProcessArticle(x_test, self.articlesDB, numSeq=self.maxSeqLen/2 - 1)
        #tokens = self.tokenizer.texts_to_sequences(texts)
        #testIDs = self.sp.encode_as_ids(' '.join(text) + " [SEP]")
        testSeq = [1] * len(testIDs)
        
        for x in self.X_train:
            #trainText = preProcessArticle(x, self.articlesDB, numSeq=self.maxSeqLen/2 - 2)
            #trainIds = self.sp.encode_as_ids("[CLS] " + ' '.join(trainText) + " [SEP]")
            
            
            traintIDs = preProcessInstance( x, self.articlesDB, 
                                            sentenceProcessor = self.sp,
                                            firstSentence = True,
                                            numSeq = self.maxSeqLen)
            trainSeq = [0] * len(traintIDs) 
            
            ids = np.concatenate([traintIDs, testIDs])
            seq =  trainSeq + testSeq 
            
            spanId = self.sp.piece_to_id("<span>")
            ids = pad_sequences([ids], maxlen=self.maxSeqLen , padding='post',truncating='post', value = spanId)
            seq = pad_sequences([seq], maxlen=self.maxSeqLen, padding='post',truncating='post', value=1)
            
            yield ids[0], seq[0]
        
    
    def fit(self, X_train, y_train):
        
        self.X_train = X_train
        self.y_train = y_train        
        
        '''
        distinctLabels = set(y_train)
        num_classes = self.num_classes = len(distinctLabels)
        
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_shape=X_train[0].shape))
        self.model.add(Dense(units=num_classes, activation='softmax'))
                
        self.model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
        
        one_hot_labels = to_categorical(y_train, num_classes=num_classes)
        
        x_matrix = np.array(X_train)
        
        self.x_matrix = x_matrix
        
        self.model.fit(x_matrix, one_hot_labels, epochs=5, batch_size=32)
        '''
        
    def predict(self, X_test, acronym):
        
        labels = []
        confidences = []
        probablities = []
        
        for x_test in X_test:            
            # DO repeat
            #repeated_x_test = np.tile(x_test, (len(self.X_train),1))
            
            # TODO
            #seq = preProcessInstance(item, self.trainArticlesDB, sentenceProcessor = self.sp, numSeq= self.maxSeqLen)
            #vect = self.albertModel(np.array([seq]))[0]
            
            probablities = [self.dualEncoder.predict((np.array([ids]), np.array([seq])))[0][0] for ids, seq in self.bertInputGenerator(x_test)]
        
            mostSimilarX =  np.argmax(np.asarray(probablities), axis=0)
            labels.append(self.y_train[mostSimilarX])
            confidences.append(probablities[mostSimilarX])
        
        
        
    
#         for x in X_test:
#             y = self.model.predict(np.array(x))[0]
# 
#             label = np.argmax(y)
#             confidence = y[label]
#             
#             labels.append(label)
#             confidences.append(confidence)
    #    x_matrix = np.array(X_test)
    #    labels = self.model.predict(x_matrix, batch_size=128)
    #    if self.num_classes > 2:
    #        print("HEre")
        
        #labels = self.classifier.predict(X_test)
        
        #decisions = self.classifier.decision_function(X_test)
        
        #confidences = self._getConfidencesFromDecisionFunction(labels, decisions)
        
        return labels,  confidences
