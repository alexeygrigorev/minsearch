"""
Mask generation for different data structures.
"""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd
from datetime import date, datetime

from .query import Condition


class Masker(ABC):
    """Abstract base class for generating filter masks."""

    @abstractmethod
    def match_mask(self, field: str, value: Any) -> np.ndarray:
        """Get mask for exact match comparisons."""

    @abstractmethod
    def range_mask(self, field: str, conditions: list[Condition]) -> np.ndarray:
        """Get mask for range comparisons."""


class PandasMasker(Masker):
    """Mask generator for pandas DataFrame data."""

    def __init__(self, data: pd.DataFrame, num_docs: int):
        self.data = data
        self.num_docs = num_docs

    def match_mask(self, field: str, value: Any) -> np.ndarray:
        if value is None:
            return self.data[field].isna().to_numpy()
        return (self.data[field] == value).to_numpy()

    def range_mask(self, field: str, conditions: list[Condition]) -> np.ndarray:
        mask = np.ones(self.num_docs, dtype=bool)
        for cond in conditions:
            op_value = cond.value
            if isinstance(op_value, (date, datetime)):
                op_value = pd.Timestamp(op_value)
            series_mask = cond.operator.func(self.data[field], op_value)
            mask = mask & series_mask.to_numpy()
        return mask.astype(float)


class DictMasker(Masker):
    """Mask generator for dict of lists data."""

    def __init__(self, data: dict, num_docs: int):
        self.data = data
        self.num_docs = num_docs

    def match_mask(self, field: str, value: Any) -> np.ndarray:
        if value is None:
            return np.array([val is None for val in self.data[field]])
        return np.array([val == value for val in self.data[field]])

    def range_mask(self, field: str, conditions: list[Condition]) -> np.ndarray:
        mask = np.ones(self.num_docs, dtype=bool)
        for cond in conditions:
            op_value = cond.value
            if isinstance(op_value, (date, datetime)):
                op_value = pd.Timestamp(op_value)
            series_mask = np.array([
                cond.operator.func(
                    pd.Timestamp(val) if isinstance(val, (date, datetime)) else val,
                    op_value
                ) if val is not None else False
                for val in self.data[field]
            ])
            mask = mask & series_mask
        return mask.astype(float)
