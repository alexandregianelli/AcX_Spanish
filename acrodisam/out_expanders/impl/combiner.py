"""
Selects the expansion whose vector obtains the closer cosine distance to the test instance
"""
from sklearn.metrics.pairwise import cosine_similarity

from helper import grouped_longest
from inputters import TrainOutDataManager
import numpy as np
from out_expanders import get_out_expander_factory

from .._base import OutExpander, OutExpanderFactory


class FactoryExpanderCombiner(OutExpanderFactory):
    def __init__(self, weights, *args, **kwargs):
        """
        :param weights
        :param args: List of text representators names and their arguments
        first list element and impair elements should contain the text representator name as string
        in the next position of the list should be the arguments to be passed to that representator
         in list or dict format.
         [name_1, args_1, name_2, args_2, ..., name_n, args_n]
         where name_x is a text representator name
         and args_n are the arguments for text representator x
        :param kwargs: to be passed to each text representator,
        includes general run information
        """
        self.weights = weights
        self.out_expander_factories = []
        for exp_name, exp_args in grouped_longest(args, 2):
            exp_args_list = [] if exp_args is None else exp_args
            factory = get_out_expander_factory(exp_name, *exp_args_list, **kwargs)
            self.out_expander_factories.append(factory)

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        # TODO - CV train expanders and get predictions

        expanders = [
            factory.get_expander(
                train_data_manager=train_data_manager,
                execution_time_observer=execution_time_observer,
            )
            for factory in self.out_expander_factories
        ]
        # Train our expander combiner -> weights by Gradient Descent
        #
        return _ExpanderCombiner(self.weights, expanders)


class _ExpanderCombiner(OutExpander):
    def __init__(self, weights, out_expanders):
        self.weights = weights
        self.out_expanders = out_expanders

    def transform(self, X):
        return [exp.transform(X) for exp in self.out_expanders]

    def fit(self, X_train, y_train):
        for i, exp in enumerate(self.out_expanders):
            exp.transform(X_train[i], y_train)

    def predict(self, X_test, acronym):

        sim_matrix = cosine_similarity(self.X_train, X_test)
        argmax_array = np.argmax(sim_matrix, axis=0)
        labels = [self.y_train[idx] for idx in argmax_array]
        confidences = sim_matrix.take(argmax_array)
        return labels, confidences
