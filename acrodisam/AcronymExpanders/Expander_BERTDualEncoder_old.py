'''

@author: jpereira
'''
import random
import os

from Logger import logging

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.utils import to_categorical

import numpy as np
#import pickle
#from keras.models import Sequential
#from keras.utils import np_utils
#from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Activation
#from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Multiply

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import params_flow as pf

import bert

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
from string_constants import FILE_LSTM_DUAL_ENCODER

from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.AcronymExpander import PredictiveExpander, FactoryPredictiveExpander

from TextRepresentators.Representator_ALBERT import preProcessInstance

import itertools

import bert

logger = logging.getLogger(__name__)

"""
def preProcessArticle(acroInstance, trainArticlesDB, numSeq = None):
        
    if isinstance(acroInstance, TrainInstance):
        term = get_expansion_without_spaces(acroInstance.expansion)

    else:
        term = acroInstance.acronym
        
    text = acroInstance.getText(trainArticlesDB)
    
    if not numSeq:
        return text_preparation.tokenizePreProcessedArticle(text.replace(term, ""))
    
    chuncks = text.lower().split(term.lower())
    preProcessedChuncks = [text_preparation.tokenizePreProcessedArticle(chunk) for chunk in chuncks]
    tokensPerChunk = [len(chunk) for chunk in preProcessedChuncks]

    if sum(tokensPerChunk) < numSeq:
        return text_preparation.tokenizePreProcessedArticle(text.replace(term, ""))
    
    
    # Find how many tokens to get from each chunk
    # TODO if slow
    #n_tokens_per_chunk, remainer_n_tokens = divmod(numSeq, (len(preProcessedChuncks) - 1) * 2)
    i = 0
    while numSeq > 0:
        if i == 0 or i == len(tokensPerChunk) - 1:
            if tokensPerChunk[i] > 0:
                tokensPerChunk[i] -=1
                numSeq -= 1
        else:
            if tokensPerChunk[i] > 1 and numSeq > 1:
                tokensPerChunk[i] -=2
                numSeq -= 2
            elif tokensPerChunk[i] > 0 and numSeq > 0:
                tokensPerChunk[i] -= 1
                numSeq -= 1       
            
        i = (i + 1) % len(tokensPerChunk)
    
    
    # Put chunks together
    finalTokens = []
    for i in range(len(tokensPerChunk)):
        tokens = preProcessedChuncks[i]
        # put all
        if tokensPerChunk[i] < 1: 
            finalTokens.extend(tokens)
        
        else:
            # first chunk
            if i == 0:
                n_tokens = len(tokens) - tokensPerChunk[i]
                finalTokens.extend(tokens[: n_tokens])
            # last chunk
            elif i == len(tokensPerChunk) - 1:
                n_tokens = len(tokens) - tokensPerChunk[i]
                finalTokens.extend(tokens[0 - n_tokens :])
            # others
            else:
                n_tokens, remainer = divmod(len(tokens) - tokensPerChunk[i], 2)
                
                left_n_tokens = n_tokens
                if remainer > 0:
                    left_n_tokens += 1
                    
                finalTokens.extend(tokens[:left_n_tokens])
                finalTokens.extend(tokens[0 - n_tokens:])
    
    return finalTokens
"""

#def preProcessInstance(acroInstance, trainArticlesDB, sentenceProcessor, numSeq = None):

class Factory_Expander_BERTDualEncoder(FactoryPredictiveExpander):

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
        self.n_epochs = 2
        
        self.datasetName = datasetName
        self.saveAndLoad = saveAndLoad
        
        self.persistentArticles = persistentArticles


        self.sp_model = tf.io.gfile.glob(os.path.join("/home/jpereira/git/AcroDisam/acrodisam_app/data/albert/albert_base", "assets/*"))[0]
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.sp_model)
        
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
                trainInstance1 = preProcessInstance(X_train[pair[0]], trainArticlesDB, sentenceProcessor, self.maxSeqLen)
                #preProcessInstance(acroInstance, trainArticlesDB, sentenceProcessor, numSeq = None)
            trainInstance2 = preProcessed.get(pair[1])
            if not trainInstance2:
                trainInstance2 = preProcessArticle(X_train[pair[1]], trainArticlesDB, self.maxSeqLen)
                
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
        tokenizer.fit_on_texts(leftTrainData)
        tokenizer.fit_on_texts(rightTrainData)

        train_c = [self.sp.encode_as_ids(' '.join(["[CLS]"] + text + ["[SEP]"])) for text in leftTrainData]
        train_r = [self.sp.encode_as_ids(' '.join(["[CLS]"] + text + ["[SEP]"])) for text in rightTrainData]

    
        #train_c = tokenizer.texts_to_sequences(leftTrainData)
        #train_r = tokenizer.texts_to_sequences(rightTrainData)
        
        #train_c = leftTrainData
        #train_r = rightTrainData
    
        if not self.maxSeqLen or self.maxSeqLen < 1:
            maxSeqLen = max([len(seq) for seq in train_c + train_r])
            logger.info("No MaxSeqLen, set to maximum: " + str(maxSeqLen))
        else:
            maxSeqLen = self.maxSeqLen
             
        #MAX_SEQUENCE_LENGTH = max([len(seq) for seq in train_c + train_r
                                                        #+ test_c + test_r
                                                        #+ dev_c + dev_r])
        
        max_number_words = len(tokenizer.word_index) + 1
        word_index = tokenizer.word_index
        
        
        logger.info("MAX_NB_WORDS: {}".format(max_number_words))
    
        train_c = pad_sequences(train_c, maxlen=maxSeqLen, padding='post',truncating='post')
        train_r = pad_sequences(train_r, maxlen=maxSeqLen, padding='post',truncating='post')
    
        # shuffle training set
        indices = np.arange(train_c.shape[0])
        
        np.random.shuffle(indices)
        train_l = trainLabels
    
        train_c = np.asarray(train_c)
        train_r = np.asarray(train_r)
        train_l = np.asarray(train_l)
    
        train_c = train_c[indices]
        train_r = train_r[indices]
        train_l = train_l[indices]        
        
        return train_c, train_r, train_l, tokenizer, maxSeqLen, max_number_words, word_index
        #TODO save into disk
        #pickle.dump([train_c, train_r, train_l], open(DATA_DIR + "train.pkl", "wb"), protocol=-1)
        #pickle.dump([test_c, test_r, test_l], open(DATA_DIR + "test.pkl", "wb"), protocol=-1)
        #pickle.dump([dev_c, dev_r, dev_l], open(DATA_DIR + "dev.pkl", "wb"), protocol=-1)

        #pickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open(DATA_DIR + "params.pkl", "wb"), protocol=-1)
    
    def getDualEncoder(self, train_c, train_r, train_l, maxSeqLen, maxNWords, word_index, embeddings="", hidden_size = 300, n_epochs = 1, executionTimeObserver=None):
        
        logger.info("Starting...")
        
        model_name = "albert_base"
        model_dir    = bert.fetch_tfhub_albert_model(model_name, "/home/jpereira/git/AcroDisam/acrodisam_app/data/albert/")
        model_params = bert.albert_params(model_name)
        l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

        # use in Keras Model here, and call model.build()

        
        #emb_dim = 300
        #hidden_size = 300
        optimizer = "adam"
        batch_size = 16
        #n_epochs = 50
        #n_epochs = 1
#        with tf.device('/cpu:0'):

        #pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
        #model3  = Model(inputs=model.input, outputs=pool_output)
        #model3.compile(loss='binary_crossentropy', optimizer=adam)
        #model3.summary()

        
        context_input_ids      = Input(shape=(maxSeqLen,), dtype='int32', name="context_input_ids")
        #context_token_type_ids = Input(shape=(maxSeqLen,), dtype='int32', name="context_token_type_ids")
        #context_output         = l_bert([context_input_ids, context_token_type_ids])
        context_output = l_bert(context_input_ids)
        context_output = keras.layers.Lambda(lambda x: x[:, 0, :])(context_output)
        
        response_input_ids      = Input(shape=(maxSeqLen,), dtype='int32', name="response_input_ids")
        #response_token_type_ids = Input(shape=(maxSeqLen,), dtype='int32', name="response_token_type_ids")
        #response_output         = l_bert([response_input_ids, response_token_type_ids])
        response_output         = l_bert(response_input_ids)
        response_output = keras.layers.Lambda(lambda x: x[:, 0, :])(response_output)

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
        concatenated = Multiply()([context_output, response_output])
        out = Dense((1), activation = "sigmoid") (concatenated)
    
        #dual_encoder = Model([context_input_ids, context_token_type_ids, response_input_ids, response_token_type_ids], out)
        dual_encoder = Model([context_input_ids, response_input_ids], out)
        
        dual_encoder.build(input_shape=(None, maxSeqLen))
        
        dual_encoder.compile(loss='binary_crossentropy',
                        optimizer=optimizer)
        
        
        
        #logger.info(l_bert.summary())
        logger.info(dual_encoder.summary())
        
        bert.load_albert_weights(l_bert, model_dir)      # should be called after model.build()
        
        #bert.load_albert_weights(l_bert, albert_dir)
    
        dual_encoder.summary()
        
        logger.info('Found %s training samples.' % len(train_c))
        
        #histories = my_callbacks.Histories()
        
        #bestAcc = 0.0
        #patience = 0 
        
        logger.info("\tbatch_size={}, nb_epoch={}".format(batch_size, n_epochs))
        if executionTimeObserver:
            executionTimeObserver.start()
        dual_encoder.fit([train_c, train_r], train_l,
                    batch_size=batch_size, epochs=n_epochs, #callbacks=[histories],
                    #validation_data=([dev_c, dev_r], dev_l),
                     verbose=1)
        if executionTimeObserver:
            executionTimeObserver.stop()
        """
        for ep in range(1, n_epochs):
                    
            dual_encoder.fit([train_c, train_r], train_l,
                    batch_size=batch_size, epochs=1, #callbacks=[histories],
                    #validation_data=([dev_c, dev_r], dev_l),
                     verbose=1)
        """
        #if save_model:
        #    print("Now saving the model... at {}".format(args.model_fname))
        #    dual_encoder.save(args.model_fname)
        
        return dual_encoder
    
            
        
    
    

        
    def getExpander(self, trainArticlesDB, acronymDB = None, articleAcronymDB = None, fold="", executionTimeObserver = None):
        
        if self.saveAndLoad:
            generatedFilesFolder = getDatasetGeneratedFilesPath(self.datasetName)
            varToName = "_".join([str(s) for s in [self.datasetName,fold, self.maxSeqLen, self.embeddings, self.hidden_size, self.n_epochs]])
            dualEncoderLocation = generatedFilesFolder + FILE_LSTM_DUAL_ENCODER + "_" + varToName + ".h5"
                
            if os.path.isfile(dualEncoderLocation):
                # TODO Load Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
                return load_model(dualEncoderLocation)    
 
        
        train_c, train_r, train_l, tokenizer, maxSeqLen, maxNWords, word_index = self.getTrainData(trainArticlesDB, 
                                                                                                   acronymDB,
                                                                                                   datasetName = self.datasetName, 
                                                                                                   fold = fold, 
                                                                                                   persistentArticles = self.persistentArticles)
        #textRepresentator = self.getRepresentator(trainArticlesDB = trainArticlesDB,
        #                                          articleAcronymDB = articleAcronymDB,
        #                                          fold = fold,
        #                                          executionTimeObserver=executionTimeObserver)
        
        dualEconder = self.getDualEncoder(train_c, train_r, train_l, maxSeqLen, maxNWords, word_index, 
                                          embeddings=self.embeddings, hidden_size = self.hidden_size, n_epochs = self.n_epochs, 
                                          executionTimeObserver=executionTimeObserver)
        
        if self.saveAndLoad:
            # TODO save Expander_DualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)
            dualEconder.save(dualEncoderLocation)
        
        return Expander_BERTDualEncoder(dualEconder, trainArticlesDB, tokenizer, maxSeqLen)


class Expander_BERTDualEncoder(PredictiveExpander):

    def __init__(self, dualEconder, articlesDB, tokenizer, maxSeqLen, expander_type=AcronymExpanderEnum.none):
        PredictiveExpander.__init__(self, expander_type, None)
        self.dualEncoder = dualEconder
        self.articlesDB = articlesDB
        self.tokenizer = tokenizer
        self.maxSeqLen = maxSeqLen
        
    def transform(self, X):
        X = super().transform(X)
        texts = [preProcessArticle(x, self.articlesDB, numSeq=self.maxSeqLen) for x in X]
        #tokens = self.tokenizer.texts_to_sequences(texts)
        tokens = [self.sp.encode_as_ids(' '.join(["[CLS]"] + text + ["[SEP]"])) for text in texts]

        seq = pad_sequences(tokens, maxlen=self.maxSeqLen, padding='post',truncating='post')
        return seq
    
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
        
        for x_test in X_test:            
            # DO repeat
            repeated_x_test = np.tile(x_test, (len(self.X_train),1))
            probablities = self.dualEncoder.predict([self.X_train, repeated_x_test])   
        
            mostSimilarX =  np.argmax(probablities, axis=0)[0]
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
