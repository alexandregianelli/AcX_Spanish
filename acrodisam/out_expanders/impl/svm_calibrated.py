'''


@author: jpereira
'''
from enum import Enum
from sklearn.model_selection import LeaveOneOut
from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from .._base import OutExpanderWithTextRepresentator, OutExpanderWithTextRepresentatorFactory

from pydantic import validate_arguments
from pydantic.types import PositiveFloat
from inputters import TrainOutDataManager

def oversampling(x, y):
    counts = Counter(y)
    for e, c in counts.items():
        if c < 2:
            indx = y.index(e)
            y.append(e)
            x.append(x[indx])
    return x, y

class FactoryExpanderSVM(OutExpanderWithTextRepresentatorFactory):
    
    class lossEnum(str, Enum):
        l1 = 'l1'
        l2 = 'l2'
    
    @validate_arguments
    def __init__(self, loss: lossEnum="l1", c:PositiveFloat=0.1, 
                 balance_class_weights:bool=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.loss = loss
        self.c = c
        self.balance_class_weights = balance_class_weights
        
    def get_expander(self, train_data_manager: TrainOutDataManager, execution_time_observer=None):
        text_representator = self._get_representator(train_data_manager=train_data_manager,
                                                  execution_time_observer=execution_time_observer)
        return _ExpanderSVM(self.loss, self.c, self.balance_class_weights, text_representator)


class _ExpanderSVM(OutExpanderWithTextRepresentator):

    def __init__(self, loss, c, balance_class_weights, text_representator):
        super().__init__(text_representator)
        self.loss = loss
        self.c = c
        self.class_weight = 'balanced' if balance_class_weights else None
        self.classifier  = None
    
    def fit(self, X_train, y_train):
        if self.loss == "l1":
            dual = False
            loss = 'squared_hinge'
        else:
            dual = True
            loss = 'squared_hinge'
        
        self.classifier = LinearSVC(C=self.c, penalty=self.loss, loss=loss, dual=dual
                                    , class_weight = self.class_weight)
        
        cv = LeaveOneOut()
        self.classifier = CalibratedClassifierCV(self.classifier, method='sigmoid', cv=cv)
        self.classifier.fit(*oversampling(list(X_train), list(y_train)))
    
    
    def predict(self, X_test, acronym):
        proba = self.classifier.predict_proba(X_test.reshape(1, -1))

        labels, confidences = self._get_labels_and_confidences_from_proba(proba)

        return labels, confidences
    
    def predict_confidences(self, text_representation, acronym, distinct_expansions):
        confidences_dict = {}
        prob_classes = self.classifier.predict_proba(text_representation.reshape(1, -1))[0]
        
        for exp, conf in zip(self.classifier.classes_, prob_classes):
            confidences_dict[exp] = conf
        
        return confidences_dict    
    