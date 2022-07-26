"""
@author: jpereira
"""
import random

from sklearn.linear_model import SGDClassifier
import numpy as np
from .._base import (
    OutExpanderWithTextRepresentator,
    OutExpanderWithTextRepresentatorFactory,
)
from .svm import FactoryExpanderSVM
from inputters import TrainOutDataManager
from pydantic.types import PositiveFloat, PositiveInt


class FactoryExpanderSVMSGD(OutExpanderWithTextRepresentatorFactory):
    def __init__(
        self,
        loss: FactoryExpanderSVM.lossEnum = "l1",
        c: PositiveFloat = 0.1,
        fixed_c: bool = False,
        iterations: PositiveInt = 1000,
        online_training: bool = False,
        balance_class_weights: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.loss = loss
        self.c = c
        self.balance_class_weights = balance_class_weights

        self.fixed_c = fixed_c
        self.iterations = iterations

        self.online_training = online_training

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        text_representator = self._get_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
        return Expander_SVM_SGD(
            self.loss,
            self.c,
            self.fixed_c,
            self.iterations,
            self.online_training,
            text_representator,
        )


class Expander_SVM_SGD(OutExpanderWithTextRepresentator):
    def __init__(
        self, loss, c, fixed_c, iterations, online_training, text_representator
    ):
        super().__init__(text_representator)
        self.loss = loss
        self.c = c
        self.fixed_c = fixed_c
        self.iterations = iterations
        self.online_training = online_training

    def fit(self, X_train, y_train):

        loss = "squared_hinge"

        n_samples = len(y_train)
        if self.fixed_c:
            alpha = 1.0 / self.c
        else:
            alpha = 1.0 / (self.c * n_samples)

        self.classifier = SGDClassifier(
            alpha=alpha, penalty=self.loss, loss=loss, max_iter=self.iterations
        )

        if self.online_training:
            shuffled_range = list(range(n_samples))
            distinct_classes = np.unique(y_train)
            for i in range(self.iterations):
                # self.classifier.partial_fit(X_train, y_train, classes=distinct_classes)
                random.shuffle(shuffled_range)
                for j in shuffled_range:
                    sample = X_train[j]
                    sample = sample.reshape(1, -1)
                    self.classifier.partial_fit(
                        sample, [y_train[j]], classes=distinct_classes
                    )

        else:
            self.classifier.fit(X_train, y_train)
        # self.classifier = LinearSVC(C=self.c, loss=self.loss)

    def predict(self, X_test, acronym):
        labels = self.classifier.predict(X_test)

        decisions = self.classifier.decision_function(X_test)

        confidences = self._getConfidencesFromDecisionFunction(labels, decisions)

        return labels, confidences
