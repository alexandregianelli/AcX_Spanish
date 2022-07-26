from AcronymExpanders import AcronymExpanderEnum
from AcronymExpanders.AcronymExpander import PredictiveExpander,FactoryPredictiveExpander
import text_preparation
from gensim.matutils import cossim
from string_constants import min_confidence
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim.models
from DataCreators import AcronymDB, ArticleDB
from string_constants import folder_doc2vecs
import logging
import gensim.models
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from helper import TrainInstance, TestInstance
from text_preparation import get_expansion_without_spaces

logger = logging.getLogger(__name__)


class Factory_Expander_Doc2VecPerAcronym(FactoryPredictiveExpander):

    def __init__(self):
        FactoryPredictiveExpander.__init__(self, None)
                
    def getExpander(self, trainArticlesDB, acronymDB = None, articleAcronymDB = None, fold="", executionTimeObserver = None):
        
        return Expander_Doc2VecPerAcronym(trainArticlesDB)


class Expander_Doc2VecPerAcronym(PredictiveExpander):
    """
    Acronym Disambiguation: A Domain Independent Approach
    """
    
    def __init__(self, articlesDB, expander_type=AcronymExpanderEnum.Doc2Vec_per_acronym):
        PredictiveExpander.__init__(self, expander_type, None)
        self.articlesDB = articlesDB
        self.cacheModelsPerAcronym = {}
        self.stop_words = set(stopwords.words('english'))
        self.p_stemmer = PorterStemmer()
        self.tokeniser = RegexpTokenizer(r'\w+')

    def transform(self, X):
        return X

    def fit(self, X_train, y_train):
        self.X_train_vec = X_train
        self.y_train_labels = y_train



    
    def _get_label(self, mostSimilarVecs):
        for similarVec in mostSimilarVecs:
            most_similar_id = similarVec[0]
            most_similar_article_id = most_similar_id.split("_")[0]

            for i, train_instance in enumerate(self.X_train_vec):
                if str(train_instance.article_id) == most_similar_article_id:
                    return self.y_train_labels[i], similarVec[1]
                
        return None, None
    
    def _get_context(self, context_ix, text, max_length):
        side_max_length = max_length // 2
        
        if context_ix == -1:
            return None
        if (side_max_length - context_ix) >= 0:
            start = 0
        else:
            start = context_ix - side_max_length
        if (side_max_length + context_ix) <= len(text):
            end = context_ix + side_max_length
        else:
            end = len(text)
        return text[start:end]
    
    def _pre_process(self, instance):
        #texts = []
        taggeddocs = []

        article_id = instance.article_id
        article_text = instance.getText(self.articlesDB)
        
        if isinstance(instance, TrainInstance):
            token = get_expansion_without_spaces(instance.expansion)
        else:
            token = instance.acronym
        
        occ_idx_list = [m.start() for m in re.finditer(token, article_text)]

        for i, context_ix in enumerate(occ_idx_list):
            para = self._get_context(context_ix,article_text, 5000)
            
            if para is None:
                continue
                
            raw = para.lower()
    
            tokens = self.tokeniser.tokenize(raw)
            stopped_tokens = [t for t in tokens if not t in self.stop_words]
    
            number_tokens = [x for x in stopped_tokens if x.isalpha]
            stemmed_tokens = [self.p_stemmer.stem(i) for i in number_tokens]
    
            #length_tokens = [i for i in stemmed_tokens if len(i) > 1]
            #texts.append(length_tokens)
            td = TaggedDocument(' '.join(stemmed_tokens).split(), [str(article_id) + "_" + str(i)])
    
            taggeddocs.append(td)
            
        return taggeddocs
    
    def _get_model(self, acronym, X_train_vec, cache=True):
        if cache and acronym in self.cacheModelsPerAcronym:
            return self.cacheModelsPerAcronym[acronym]

        taggeddocs = []
        for x in X_train_vec:
            taggeddocs = taggeddocs + self._pre_process(x)
        
        model = gensim.models.Doc2Vec(taggeddocs, dm=1, alpha=0.025, vector_size=500, min_alpha=0.025, min_count=0)
        print("check 2")
        for epoch in range(15):
            model.train(taggeddocs, total_examples=model.corpus_count, epochs=model.iter)
            model.min_alpha = model.alpha
            
        if cache:
            self.cacheModelsPerAcronym[acronym] = model
            
        return model
    
    def _get_most_frequent_label(self, labels, confidences):
        if len(labels) == 1:
            return labels[0], confidences[0]
        
        frequencies = {}
        accConfidences = {}
        for (label, confidence) in zip(labels, confidences):
            if label not in frequencies:
                frequencies[label] = 0
            frequencies[label] = frequencies[label] + 1
            
            if label not in accConfidences:
                accConfidences[label] = 0
            accConfidences[label] = accConfidences[label] + confidence
            
        label = max(frequencies, key=lambda key: frequencies[key])
        # We average for now
        confidence = accConfidences[label] / frequencies[label]
        
        return label, confidence
    
    def predict(self, X_test, acronym):
        
        model = self._get_model(acronym, self.X_train_vec, cache=True)
        
        labels = []
        confidences = []
        for test_instance in X_test:
            instanceLabels = []
            instanceConfidences = [] 
            taggeddocs = self._pre_process(test_instance)
            for taggedDoc in taggeddocs:
                document_words = taggedDoc.words
                try:
                    test_vect = model.infer_vector(document_words)
                    mostSimilarVecs = model.docvecs.most_similar([test_vect])
                except TypeError as err:
                    logger.error("article_id=" + str(test_instance.article_id) + " acronym=" + str(acronym), err)
                    logger.error(str(self.y_train_labels))
                except  KeyError as err:
                    logger.error("article_id=" + str(test_instance.article_id) + " acronym=" + str(acronym), err)
                    logger.error(str(self.y_train_labels))
     #       most_similar_article_id = mostSimilarVecs[0][0]
     #       confidences.append(mostSimilarVecs[0][1])
            
     #       label = None
            #find expansion label
     #       i=0
      #      for train_instance in self.X_train_vec:
      #          if train_instance.article_id == most_similar_article_id:
      #              label = self.y_train_labels[i]    
      #          i += 1
      
                [label, confidence] = self._get_label(mostSimilarVecs)
                instanceLabels.append(label)
                instanceConfidences.append(confidence)
            
            [finalLable, finalConfidence] = self._get_most_frequent_label(instanceLabels, instanceConfidences)
            labels.append(finalLable)
            confidences.append(finalConfidence)
            
            if label == None:
                logger.error("Label = None")

            
        return labels, confidences
    
                    
        
        
        
        
        
        