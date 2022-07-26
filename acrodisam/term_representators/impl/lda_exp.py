from typing import Union, Optional

from gensim.matutils import sparse2full
from pydantic.decorator import validate_arguments
from typing_extensions import Literal

from DataCreators.LDAModel import create_model
from DataCreators.LDAModel import preProcessText
from acronym_expander import RunConfig
from helper import ExecutionTimeObserver
import numpy as np

from .._base import (
    TermRepresentatorAcronymIndependent,
    TermRepresentatorFactory,
    TermRepresentator,
)
from inputters import TrainOutDataManager, InputArticle


# Representator based on Jacobs et al. Acronym: identification, expansion and disambiguation
# Originally applied for texts in Hebrew language
class FactoryLDAEXP(TermRepresentatorFactory):
    """
    Text representator factory to create LDA models
    """

    @validate_arguments
    def __init__(
        self,
        epochs: int = 1,
        num_topics: Union[Literal["log(nub_distinct_words)+1"], int] = 100,
        run_config: Optional[RunConfig] = RunConfig(),
    ):
        """

        :param epochs: Number of passes (epochs) through the corpus during training default=1
        :param num_topics: The number of requested latent topics to be extracted from the training
         corpus, default=100
        :param run_config: general run configurations
        """

        self.epochs = epochs
        self.num_topics = num_topics

        self.dataset_name = run_config.name
        self.save_and_load = run_config.save_and_load
        self.persistent_articles = run_config.persistent_articles

    def get_term_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TermRepresentator:

        lda_model = create_model(
            process_pool=None,
            datasetName=self.dataset_name,
            articleDB=train_data_manager.get_preprocessed_articles_db(),
            num_topics=self.num_topics,
            numPasses=self.epochs,
            fold=train_data_manager.get_fold(),
            saveAndLoad=self.save_and_load,
            persistentArticles=self.persistent_articles,
            executionTimeObserver=execution_time_observer,
        )
        return _RepresentatorLDAEXP(lda_model)


class _RepresentatorLDAEXP(TermRepresentatorAcronymIndependent):
    """
    take LDA vectors of labelled articles
    """

    def __init__(self, ldaModelAll):
        super().__init__()
        self.ldaModel = ldaModelAll.ldaModel
        self.dictionary = ldaModelAll.dictionary
        self.articleIDToLDADict = ldaModelAll.articleIDToLDADict

    def _transform_expansion_term(self, expansion):
        vector_size = self.ldaModel.num_topics
        expansionVector = np.zeros(vector_size)

        expansionTokens = preProcessText(expansion)
        tokenIDs = self.dictionary.doc2idx(expansionTokens)
        for tokenId in tokenIDs:
            lda_vector = self.ldaModel.get_term_topics(tokenId)
            if lda_vector:
                expansionVector = expansionVector + self._getDenseVector(lda_vector)

        return expansionVector

    def _transform_input_text(self, article: InputArticle):

        cleaned_words = preProcessText(article.get_preprocessed_text())
        bow = self.dictionary.doc2bow(cleaned_words)
        lda_vector = self.ldaModel[bow]

        return self._getDenseVector(lda_vector)

    def _getDenseVector(self, sparse_vec):
        return sparse2full(sparse_vec, self.ldaModel.num_topics)

    def close(self):
        # TODO Close sqlite stuff bow?
        print("TODO closing")

    def __del__(self):
        self.close()
