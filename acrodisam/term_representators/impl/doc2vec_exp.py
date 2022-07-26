"""
Expansion embeddings based on Doc2vec term representator, each document/text is labeled
with the existing expansions. This is different from most common doc2vec which assigns a
unique identifier as labels to distiguish between documents
"""
from typing import Optional

from pydantic import validate_arguments, PositiveInt
from typing_extensions import Literal

from DataCreators.Doc2VecModel import trainDoc2VecModel, preProcessText
from acronym_expander import RunConfig
from helper import ExecutionTimeObserver
from inputters import TrainOutDataManager, InputArticle
from term_representators._base import TermRepresentator

from .._base import TermRepresentatorAcronymIndependent, TermRepresentatorFactory


class FactoryDoc2VecExp(
    TermRepresentatorFactory
):  # pylint: disable=too-few-public-methods
    """
    Creates doc2vec models that learn to assign embeddings to expansions instead of documents
    """

    @validate_arguments
    def __init__(  # pylint: disable=too-many-arguments
        self,
        epoch: PositiveInt = 50,
        algorithm: Literal["Skip-gram", "CBOW"] = "CBOW",
        vector_size: PositiveInt = 200,
        window_size: PositiveInt = 8,
        run_config: Optional[RunConfig] = RunConfig(),
    ):
        """

        :param epoch: Number of iterations (epochs) over the corpus
        :param algorithm: Defines the training algorithm
        :param vector_size: Dimensionality of the feature vectors
        :param window_size: The maximum distance between the current
        and predicted word within a sentence
        :param run_config: general run configurations
        """

        # self._set_default_values(run_config.name)

        self.epoch = epoch
        if algorithm == "Skip-gram":
            self.algorithm = 1
        elif algorithm == "CBOW":
            self.algorithm = 0
        else:
            raise TypeError("Algorithm value not known: %s" % str(algorithm))

        self.vector_size = vector_size
        self.window_size = window_size

        self.run_config = run_config

    def get_term_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ) -> TermRepresentator:

        doc2vec_model, _ = trainDoc2VecModel(
            articleDB=train_data_manager.get_preprocessed_articles_db(),
            articleAcronymDB=train_data_manager.get_article_acronym_db(),
            expansionAsTags=True,
            epochs=self.epoch,
            dm=self.algorithm,
            vector_size=self.vector_size,
            window=self.window_size,
            datasetName=self.run_config.name,
            fold=train_data_manager.get_fold(),
            saveAndLoad=self.run_config.save_and_load,
            persistentArticles=self.run_config.persistent_articles,
            executionTimeObserver=execution_time_observer,
        )

        return _RepresentatorDoc2VecExp(doc2vec_model)


class _RepresentatorDoc2VecExp(
    TermRepresentatorAcronymIndependent
):  # pylint: disable=too-few-public-methods
    def __init__(self, doc2vec_model):
        super().__init__()
        self.doc2vec_model = doc2vec_model

    def _transform_input_text(self, article: InputArticle):
        preprocessed_input = preProcessText(article.get_preprocessed_text())
        vect = self.doc2vec_model.infer_vector(preprocessed_input)
        return vect

    def _transform_expansion_term(self, expansion):
        vect = self.doc2vec_model[expansion]
        return vect
