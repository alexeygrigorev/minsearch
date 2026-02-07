"""
Filter class for applying validated query objects.
"""
import numpy as np
import pandas as pd
from datetime import date, datetime

from .query import (
    KeywordQuery,
    NumericExactQuery,
    NumericRangeQuery,
    DateExactQuery,
    DateRangeQuery,
    ValidatedFilter,
    Condition,
    Operator,
)
from .masker import Masker, PandasMasker, DictMasker
from .validator import Validator, FilterValidationError


class Filter:
    """
    Handles filtering for keyword, numeric, and date fields.
    """

    def __init__(
        self,
        keyword_fields: list[str],
        numeric_fields: list[str],
        date_fields: list[str],
        keyword_data: dict[str, list] | pd.DataFrame,
        numeric_data: dict[str, list] | pd.DataFrame,
        date_data: dict[str, list] | pd.DataFrame,
        num_docs: int,
    ):
        self.validator = Validator(keyword_fields, numeric_fields, date_fields)
        self.num_docs = num_docs

        self.keyword_data = keyword_data
        self.numeric_data = numeric_data
        self.date_data = date_data

    def _create_masker(self, data: dict | pd.DataFrame) -> Masker:
        """Create masker based on data type."""
        if isinstance(data, pd.DataFrame):
            return PandasMasker(data, self.num_docs)
        else:
            return DictMasker(data, self.num_docs)

    def refresh(
        self,
        keyword_data: dict | pd.DataFrame | None = None,
        numeric_data: dict | pd.DataFrame | None = None,
        date_data: dict | pd.DataFrame | None = None,
        num_docs: int | None = None,
    ) -> None:
        """Refresh the filter with updated data references."""
        if keyword_data is not None:
            self.keyword_data = keyword_data
        if numeric_data is not None:
            self.numeric_data = numeric_data
        if date_data is not None:
            self.date_data = date_data
        if num_docs is not None:
            self.num_docs = num_docs

    def apply(self, filter_dict: dict) -> np.ndarray:
        """Apply filters and return a boolean mask (1 for pass, 0 for fail)."""
        if not filter_dict:
            return np.ones(self.num_docs, dtype=float)

        validated = self.validator.validate(filter_dict)
        return self._apply_validated(validated)

    def _apply_validated(self, validated: ValidatedFilter) -> np.ndarray:
        """Apply a validated filter object and return a boolean mask."""
        mask = np.ones(self.num_docs, dtype=float)

        mask = mask * self._apply_keyword(validated.keyword_queries, mask)
        mask = mask * self._apply_numeric(validated.numeric_queries, mask)
        mask = mask * self._apply_date(validated.date_queries, mask)

        return mask

    def _apply_keyword(self, queries: list[KeywordQuery], mask: np.ndarray) -> np.ndarray:
        masker = self._create_masker(self.keyword_data)
        for query in queries:
            field_mask = masker.match_mask(query.field, query.value)
            mask = mask * field_mask
        return mask

    def _apply_numeric(self, queries: list[NumericExactQuery | NumericRangeQuery], mask: np.ndarray) -> np.ndarray:
        masker = self._create_masker(self.numeric_data)
        for query in queries:
            if isinstance(query, NumericRangeQuery):
                field_mask = masker.range_mask(query.field, query.conditions)
            else:
                field_mask = masker.match_mask(query.field, query.value)
            mask = mask * field_mask
        return mask

    def _apply_date(self, queries: list[DateExactQuery | DateRangeQuery], mask: np.ndarray) -> np.ndarray:
        masker = self._create_masker(self.date_data)
        for query in queries:
            if isinstance(query, DateRangeQuery):
                field_mask = masker.range_mask(query.field, query.conditions)
            else:
                field_mask = masker.match_mask(query.field, query.value)
            mask = mask * field_mask
        return mask
