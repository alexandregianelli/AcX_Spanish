from collections import Counter
from typing import Optional

from sklearn.feature_extraction import DictVectorizer

from helper import TrainInstance, ExecutionTimeObserver
from inputters import TrainOutDataManager, InputArticle
import numpy as np
from run_config import RunConfig
from text_preparation import get_expansion_without_spaces
import text_preparation

from .._base import TextRepresentatorFactory, TextRepresentator


# from nltk.tokenize import word_tokenize
word_tokenize = text_preparation.tokenizePreProcessedArticle


def tokens_to_ngrams(tokens, n):
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngrams_split(ngrams, text):
    tokens = word_tokenize(text)
    result = []
    for n in ngrams:
        result += tokens_to_ngrams(tokens, n)

    return Counter(result)


def ngrams_split_weights(ngrams, weights, text):
    tokens = word_tokenize(text)
    result = []
    for n, w in zip(ngrams, weights):
        result.append(applyWeights(Counter(tokens_to_ngrams(tokens, n)), w))

    return mergeDicts(result)


def mergeDicts(dicts):
    mergedDict = {}
    for d in dicts:
        for k, v in d.items():
            old = mergedDict.setdefault(k, 0.0)
            mergedDict[k] = old + v
    return mergedDict


def applyWeights(counter, weight):
    newDict = {}
    for k, v in counter.items():
        newDict[k] = v * weight

    return newDict


def trainArticlesWithoutExpansion(trainArticlesDB, articleAcronymDB):
    for articleId, text in trainArticlesDB.items():
        textWitoutExp = text
        acronymExpansions = articleAcronymDB.get(articleId)
        if acronymExpansions:
            for expansion in acronymExpansions.values():
                concept = get_expansion_without_spaces(expansion)
                textWitoutExp = textWitoutExp.replace(concept, "")

        yield textWitoutExp


class FactoryNGramsContextVector(TextRepresentatorFactory):
    def __init__(
        self, ngrams=["1"], weights=["1"], run_config: Optional[RunConfig] = RunConfig()
    ):
        self.ngrams = [int(n) for n in ngrams]
        self.weights = [float(w) for w in weights]

        self.ngrams, self.weights = zip(
            *[(n, w) for n, w in zip(self.ngrams, self.weights) if w != 0.0]
        )

    #    pass

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ):
        vocabulary = DictVectorizer()

        # discover corpus and vectorize file word frequencies in a single pass
        if execution_time_observer:
            execution_time_observer.start()

        trainArticles = trainArticlesWithoutExpansion(
            train_data_manager.get_preprocessed_articles_db(),
            train_data_manager.get_article_acronym_db(),
        )

        result_matrix = vocabulary.fit_transform(
            ngrams_split(self.ngrams, f) for f in trainArticles
        )
        # get the maximum number of occurrences of the same word in the corpus, this is for normalization purposes
        max_c = result_matrix.sum(axis=1).max()

        if execution_time_observer:
            execution_time_observer.stop()
        return _NGramsContextVector(
            self.ngrams,
            self.weights,
            train_data_manager.get_preprocessed_articles_db(),
            vocabulary,
            float(max_c),
        )


class _NGramsContextVector(TextRepresentator):
    """
    take doc2vec vectors of labelled articles
    """

    def __init__(self, ngrams, weights, articles_db, vocabulary, max_c):
        super().__init__()
        self.ngrams = ngrams
        self.weights = weights
        self.vocabulary = vocabulary
        self.articles_db = articles_db
        self.max_c = max_c

    def tranform_test_instance(self, acronym_list, article: InputArticle):
        text = article.get_preprocessed_text()
        for acronym in acronym_list:
            yield self._transform_instance(acronym, text)

    def _transform_train_instance(self, train_instance: TrainInstance):
        concept = get_expansion_without_spaces(train_instance.expansion)
        text = train_instance.getText(self.articles_db)
        return self._transform_instance(concept, text)

    def _transform_instance(self, concept, text):
        """
        Transforms an instance into a document context representation
        :param instance_x: train or test instance representing
         a document with an expansion or acronym
        """
        # tokens = preprocessed_text_tokenizer(text.replace(concept, ""))
        text = text.replace(concept, "")

        return np.divide(
            self.vocabulary.transform(
                ngrams_split_weights(self.ngrams, self.weights, text)
            ).toarray()[0],
            self.max_c,
        )
