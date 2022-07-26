"""

@author: jpereira
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import numpy as np

from inputters import TrainOutDataManager
from .._base import (
    OutExpanderWithTextRepresentator,
    OutExpanderWithTextRepresentatorFactory,
)


class FactoryExpanderKeras(OutExpanderWithTextRepresentatorFactory):
    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        text_representator = self._get_representator(
            train_data_manager=train_data_manager,
            execution_time_observer=execution_time_observer,
        )
        return _ExpanderKeras(text_representator)


class _ExpanderKeras(OutExpanderWithTextRepresentator):
    def fit(self, X_train, y_train):

        distinctLabels = set(y_train)
        num_classes = self.num_classes = len(distinctLabels)

        self.model = Sequential()
        self.model.add(Dense(units=64, activation="relu", input_shape=X_train[0].shape))
        self.model.add(Dense(units=num_classes, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
        )

        one_hot_labels = to_categorical(y_train, num_classes=num_classes)

        x_matrix = np.array(X_train)

        self.x_matrix = x_matrix

        self.model.fit(x_matrix, one_hot_labels, epochs=5, batch_size=32)

    def predict(self, X_test, acronym):
        x_matrix = np.array(X_test)
        y = self.model.predict(x_matrix)

        labels = np.argmax(y, axis=1)
        confidences = np.max(y, axis=1)
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

        # labels = self.classifier.predict(X_test)

        # decisions = self.classifier.decision_function(X_test)

        # confidences = self._getConfidencesFromDecisionFunction(labels, decisions)

        return labels, confidences
