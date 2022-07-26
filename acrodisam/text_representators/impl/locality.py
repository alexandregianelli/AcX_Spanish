from collections import Counter
import collections
from typing import Optional

from nltk.tokenize import RegexpTokenizer, sent_tokenize
from pydantic.decorator import validate_arguments
from pydantic.types import confloat
from sklearn.feature_extraction import DictVectorizer

from Logger import logging
from helper import TrainInstance, ExecutionTimeObserver
from inputters import TrainOutDataManager
import numpy as np
from run_config import RunConfig
from string_constants import (
    SCIENCE_WISE_DATASET,
    CS_WIKIPEDIA_DATASET,
    FULL_WIKIPEDIA_DATASET,
    MSH_ORGIN_DATASET,
)
from text_preparation import (
    get_expansion_without_spaces,
    word_tokenizer_and_transf,
    p_stemmer,
    preprocessed_text_tokenizer,
)

from .._base import TextRepresentator, TextRepresentatorFactory


logger = logging.getLogger(__name__)

regex_word_tokenizer = lambda text: RegexpTokenizer(r"\w+").tokenize(text.lower())
unit_interval = confloat(ge=0, le=1)


class FactoryLocality(TextRepresentatorFactory):
    @validate_arguments
    def __init__(
        self,
        sentence_decay: unit_interval = 0.0,
        paragraph_decay: unit_interval = 0.0,
        paragraph_distance=None,
        vectorizer_factory=None,
        run_config: Optional[RunConfig] = RunConfig(),
    ):

        self.sentence_decay = sentence_decay
        self.paragraph_decay = paragraph_decay
        self.paragraph_distance = paragraph_distance
        self.vectorizer_factory = vectorizer_factory

        self.dataset_name = run_config.name
        self.save_and_load = run_config.save_and_load

        if SCIENCE_WISE_DATASET.casefold() in self.dataset_name.casefold():
            self.word_tokenizer = word_tokenizer_and_transf
        elif MSH_ORGIN_DATASET.casefold() in self.dataset_name.casefold():
            self.word_tokenizer = word_tokenizer_and_transf
        elif CS_WIKIPEDIA_DATASET.casefold() in self.dataset_name.casefold():
            self.word_tokenizer = lambda text: word_tokenizer_and_transf(
                text, word_transf_func=p_stemmer.stem
            )
        elif FULL_WIKIPEDIA_DATASET in self.dataset_name.casefold():
            self.word_tokenizer = lambda text: word_tokenizer_and_transf(
                text, word_transf_func=p_stemmer.stem
            )
        else:
            logger.warning(
                "No preprocessing found in Locality for dataset: %s", self.dataset_name
            )
            logger.warning("Using full preprocessing")
            self.word_tokenizer = lambda text: word_tokenizer_and_transf(
                text, word_transf_func=p_stemmer.stem
            )

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TextRepresentator:
        vectorizer = None

        if self.vectorizer_factory is None:
            vocabulary = DictVectorizer()

            # discover corpus and vectorize file word frequencies in a single pass
            if execution_time_observer:
                execution_time_observer.start()

            result_matrix = vocabulary.fit_transform(
                Counter(preprocessed_text_tokenizer(f))
                for f in train_data_manager.get_preprocessed_articles_db().values()
            )
            # get the maximum number of occurrences of the same word in a document, this is for normalization purposes
            max_c = result_matrix.max()

            if execution_time_observer:
                execution_time_observer.stop()
        else:
            vectorizer = self.vectorizer_factory.getVectorizer(
                train_data_manager, execution_time_observer
            )
            vocabulary = DictVectorizer()
            vocabulary = vocabulary.fit([Counter(vectorizer.get_feature_names())])

        if self.paragraph_distance:
            return _LocalityParagDist(
                train_data_manager.articles_raw_db,
                self.word_tokenizer,
                vocabulary,
                vectorizer,
                sentence_decay=self.sentence_decay,
                paragraph_decay=self.paragraph_decay,
                paragraph_distance=self.paragraph_distance,
                max_c=float(max_c),
            )

        return _Locality(
            train_data_manager.articles_raw_db,
            self.word_tokenizer,
            vocabulary,
            vectorizer,
            sentence_decay=self.sentence_decay,
            paragraph_decay=self.paragraph_decay,
            max_c=float(max_c),
        )
        # contextVectorMode = self.contextVectorMode)


class _Locality(TextRepresentator):
    def __init__(
        self,
        articles_db,
        word_tokenizer,
        vocabulary,
        vectorizer=None,
        sentence_decay=0.7,
        paragraph_decay=0.4,
        max_c=1,
    ):
        self.articles_db = articles_db
        # self.cacheModelsPerAcronym = {}
        # self.preprocessing = preprocessing
        # self.stop_words = set(stopwords.words('english'))
        # self.p_stemmer = PorterStemmer()
        self.pragraph_tokenizer = lambda text: text.split("\n")
        self.sent_tokenizer = sent_tokenize
        self.word_tokenizer = word_tokenizer
        self.sentence_decay = sentence_decay
        self.paragraph_decay = paragraph_decay
        self.vocabulary = vocabulary
        self.vectorizer = vectorizer
        self.max_c = max_c

        # if contextVectorMode:
        #    self.transform = self.transformContextVector
        # else:
        # self.transform = self.transformDefault

    def _addWeightToTokens(self, weight, tokens, weightedTokens):
        for token in tokens:
            # We always choose the highest possible weight
            oldWeight = weightedTokens.get(token, 0.0)

            if weight > oldWeight:
                weightedTokens[token] = weight

    def _weightTokensInSentences(self, sent_list, weightedTokens):
        for i, sent in enumerate(sent_list):
            tokens = self.word_tokenizer(sent)
            if i == 0:  # Same sentence more weight
                weight = 1.0 - (1.0 - self.paragraph_decay)  # this is equivalent to 1
            else:
                weight = (1.0 - self.sentence_decay) - (
                    1.0 - self.paragraph_decay
                )  # this is equivalent to same paragraph, different sentence

            self._addWeightToTokens(weight, tokens, weightedTokens)

    def transformText(self, text):
        # make sure we don't collect words with punctuation
        tokens = self.word_tokenizer(text)
        if self.vectorizer:
            return self.vectorizer.transform([" ".join(tokens)]).toarray()[0]

        return self.vocabulary.transform(Counter(tokens)).toarray()[0]

    def getWeightedTokens(self, text, concept, weightedTokens=None):
        if weightedTokens == None:
            weightedTokens = {}

        conceptsSurround = text.split(concept)

        # First string as only an acronym after it
        paragraphs = self.pragraph_tokenizer(conceptsSurround[0])

        # We reverse because the function expects that the first paragraph are closer to the concept
        # paragraphs.reverse()
        # We just want the same paragrpah
        sentences = self.sent_tokenizer(paragraphs[-1])
        sentences.reverse()
        # send sentences by order - first sentence has an acronym/exp occurance
        self._weightTokensInSentences(sentences, weightedTokens)

        for surroundText in conceptsSurround[1:-1]:
            paragraphs = self.pragraph_tokenizer(surroundText)

            firstPSentences = self.sent_tokenizer(paragraphs[0])
            self._weightTokensInSentences(firstPSentences, weightedTokens)

            # Check if we have more than one paragraph
            if len(paragraphs) < 2:
                # We check if the first and only paragraph has more then one sentence
                if len(firstPSentences) > 1:
                    # if so we update the weights of the last sentence only
                    self._weightTokensInSentences([firstPSentences[-1]], weightedTokens)

            else:
                # Last paragraph also contains an acronym/exp
                lastPSentences = self.sent_tokenizer(paragraphs[-1])
                # In this case, the acronym/exp belongs to the last sentence
                lastPSentences.reverse()
                self._weightTokensInSentences(lastPSentences, weightedTokens)

        # Last string as only an acronym before it
        lastParagraphs = self.pragraph_tokenizer(conceptsSurround[-1])

        # We just want the same paragraph
        sentences = self.sent_tokenizer(lastParagraphs[0])
        self._weightTokensInSentences(sentences, weightedTokens)

        return weightedTokens

    def applyWeights(self, text, weightedTokens):
        transf = self.transformText(text)
        # transf = self.vocabulary.transform(Counter(preprocessed_tokens)).toarray()[0]
        # Lets apply a paragraph decay to all
        # transf = self.paragraphDecay * transf
        # np.ones(transf.shape) * self.paragraphDecay
        transfWeightedTokens = self.vocabulary.transform(weightedTokens).toarray()
        newTransf = (
            np.ones(transf.shape) * (1 - self.paragraph_decay) + transfWeightedTokens
        ) * transf

        return newTransf[0]

    def _getTextAndConcept(self, x):
        if isinstance(x, TrainInstance):
            concept = get_expansion_without_spaces(x.expansion)

        else:
            concept = x.acronym

        text = x.getText(self.articles_db)

        return text, concept

    def _transform_instance(self, concept, raw_text):

        weightedTokens = self.getWeightedTokens(raw_text, concept)

        # return self.applyWeights(text.replace(concept,''), weightedTokens)
        vector = self.applyWeights(raw_text, weightedTokens)

        # tokens = preprocessed_text_tokenizer(text.replace(concept, ""))

        return np.divide(vector, self.max_c)

    def tranform_test_instance(self, acronym_list, article):
        text = article.get_raw_text()
        for acronym in acronym_list:
            yield self._transform_instance(acronym, text)

    def _transform_train_instance(self, train_instance):
        concept = get_expansion_without_spaces(train_instance.expansion)
        text = train_instance.getText(self.articles_db)
        return self._transform_instance(concept, text)


class TokenMapping(collections.abc.Mapping):
    def __init__(self, weightDict):
        self.weightDict = weightDict

    def __getitem__(self, key):
        value = self.weightDict.get(key)
        if value is None:
            return None
        return value[0] * value[1]

    def __iter__(self):
        return self.weightDict.__iter__()

    def __len__(self):
        return len(self.weightDict)


class _LocalityParagDist(_Locality):
    def __init__(
        self,
        articles_db,
        word_tokenizer,
        vocabulary,
        vectorizer=None,
        sentence_decay=0.7,
        paragraph_decay=0.4,
        paragraph_distance=3,
        max_c=1,
    ):
        super().__init__(
            articles_db=articles_db,
            word_tokenizer=word_tokenizer,
            vocabulary=vocabulary,
            vectorizer=vectorizer,
            sentence_decay=sentence_decay,
            paragraph_decay=paragraph_decay,
            max_c=float(max_c),
        )

        self.paragraph_distance = paragraph_distance

    def _addWeightToTokens(self, weight, tokens, weightedTokens):
        for token in tokens:
            out = weightedTokens.setdefault(token, [0.0, 0])
            oldWeight = out[0]
            out[1] += 1

            # We always choose the highest possible weight
            if weight > oldWeight:
                out[0] = weight

    def _weightTokensInSentences(self, sent_list, weightedTokens):
        for i, sent in enumerate(sent_list):
            tokens = self.word_tokenizer(sent)
            if i == 0:  # Same sentence more weight
                weight = 1.0  # this is equivalent to 1
            else:
                weight = (
                    1.0 - self.sentence_decay
                )  # this is equivalent to same paragraph, different sentence

            self._addWeightToTokens(weight, tokens, weightedTokens)

    def _weightsTokensInParagraphs(self, paragraphs, betweenConcepts, weightedTokens):
        if betweenConcepts:
            if len(paragraphs) > self.paragraph_distance * 2:
                paragraphs = (
                    paragraphs[: self.paragraph_distance]
                    + paragraphs[-self.paragraph_distance :]
                )
        else:
            if len(paragraphs) > self.paragraph_distance:
                paragraphs = paragraphs[: self.paragraph_distance]

        for para in paragraphs:
            sentences = self.sent_tokenizer(para)
            for sent in sentences:
                tokens = self.word_tokenizer(sent)
                self._addWeightToTokens(
                    1.0 - self.paragraph_decay, tokens, weightedTokens
                )

    def getWeightedTokens(self, text, concept, weightedTokens=None):
        if weightedTokens == None:
            weightedTokens = {}

        conceptsSurround = text.split(concept)

        # First string as only an acronym after it
        paragraphs = self.pragraph_tokenizer(conceptsSurround[0])

        paramParagraphs = paragraphs[:-1]
        paramParagraphs.reverse()
        self._weightsTokensInParagraphs(paramParagraphs, False, weightedTokens)
        # We reverse because the function expects that the first paragraph are closer to the concept
        # paragraphs.reverse()
        # We just want the same paragrpah
        sentences = self.sent_tokenizer(paragraphs[-1])
        sentences.reverse()
        # send sentences by order - first sentence has an acronym/exp occurance
        self._weightTokensInSentences(sentences, weightedTokens)

        for surroundText in conceptsSurround[1:-1]:
            paragraphs = self.pragraph_tokenizer(surroundText)

            firstPSentences = self.sent_tokenizer(paragraphs[0])

            # Check if we have more than one paragraph
            if len(paragraphs) < 2:
                # We check if the first and only paragraph has more then one sentence
                if len(firstPSentences) > 1:
                    # if so we update the weights of the last sentence only
                    self._weightTokensInSentences(firstPSentences[:-1], weightedTokens)
                    self._weightTokensInSentences([firstPSentences[-1]], weightedTokens)
                else:
                    self._weightTokensInSentences(firstPSentences, weightedTokens)

            else:
                self._weightTokensInSentences(firstPSentences, weightedTokens)

                # Last paragraph also contains an acronym/exp
                lastPSentences = self.sent_tokenizer(paragraphs[-1])
                # In this case, the acronym/exp belongs to the last sentence
                lastPSentences.reverse()
                self._weightTokensInSentences(lastPSentences, weightedTokens)

                if len(paragraphs) > 2:
                    self._weightsTokensInParagraphs(
                        paragraphs[1:-1], True, weightedTokens
                    )

        # Last string as only an acronym before it
        lastParagraphs = self.pragraph_tokenizer(conceptsSurround[-1])

        self._weightsTokensInParagraphs(lastParagraphs[1:], False, weightedTokens)

        # We just want the same paragraph
        sentences = self.sent_tokenizer(lastParagraphs[0])
        self._weightTokensInSentences(sentences, weightedTokens)

        return weightedTokens

    def _transform_instance(self, concept, raw_text):
        weightedTokens = self.getWeightedTokens(raw_text, concept)

        transf_weighted_tokens = self.vocabulary.transform(
            [TokenMapping(weightedTokens)]
        ).toarray()[0]
        return np.divide(transf_weighted_tokens, self.max_c)
