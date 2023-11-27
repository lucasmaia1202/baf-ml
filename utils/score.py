"""Module to calculate the scores of the model."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Score(ABC):
    """Class to store model scores."""

    @abstractmethod
    def calculate(self, y_true: Any, y_pred: Any) -> None:
        """Method to calculate the scores."""


@dataclass
class ScoreClassification(Score):
    """Class to store model scores."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    kappa: Optional[float] = None
    confusion_matrix: Optional[Any] = None

    def calculate(self, y_true: Any, y_pred: Any) -> None:
        """Method to calculate the scores."""
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred)
        self.recall = recall_score(y_true, y_pred)
        self.f1 = f1_score(y_true, y_pred)
        self.auc_roc = roc_auc_score(y_true, y_pred)
        self.kappa = cohen_kappa_score(y_true, y_pred)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
