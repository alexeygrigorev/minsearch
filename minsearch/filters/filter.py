"""
Filter class for applying validated query objects.
"""
import numpy as np
import pandas as pd

from .query import (
    KeywordQuery,
    NumericExactQuery,
    NumericRangeQuery,
    DateExactQuery,
    DateRangeQuery,
    ValidatedFilter,
)
from .masker import Masker, PandasMasker, DictMasker
from .validator import Validator
from .field import FieldData


class Filter:
    """
    Handles filtering for keyword, numeric, and date fields.
    """

    def __init__(
        self,
        keyword: FieldData,
        numeric: FieldData,
        date: FieldData,
        num_docs: int | None = None,
    ):
        self.keyword = keyword
        self.numeric = numeric
        self.date = date
        # Use provided num_docs, or fall back to keyword data size
        self.num_docs = num_docs if num_docs is not None else keyword.num_docs

        self.validator = Validator(keyword.fields, numeric.fields, date.fields)

    def _create_masker(self, data: dict | pd.DataFrame) -> Masker:
        """Create masker based on data type."""
        # Calculate num_docs from the actual data
        if isinstance(data, pd.DataFrame):
            num_docs = len(data)
            return PandasMasker(data, num_docs)
        else:
            num_docs = len(next(iter(data.values()))) if data else 0
            return DictMasker(data, num_docs)

    def refresh(
        self,
        keyword_data: dict | pd.DataFrame | None = None,
        numeric_data: dict | pd.DataFrame | None = None,
        date_data: dict | pd.DataFrame | None = None,
        num_docs: int | None = None,
    ) -> None:
        """Refresh the filter with updated field data."""
        if keyword_data is not None:
            self.keyword = FieldData(fields=self.keyword.fields, data=keyword_data)
        if numeric_data is not None:
            self.numeric = FieldData(fields=self.numeric.fields, data=numeric_data)
        if date_data is not None:
            self.date = FieldData(fields=self.date.fields, data=date_data)
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
        # Determine num_docs from the first non-empty data source
        num_docs = self._get_num_docs()

        mask = np.ones(num_docs, dtype=float)

        mask = mask * self._apply_keyword(validated.keyword_queries, mask)
        mask = mask * self._apply_numeric(validated.numeric_queries, mask)
        mask = mask * self._apply_date(validated.date_queries, mask)

        return mask

    def _get_num_docs(self) -> int:
        """Get the number of documents from the first non-empty data source."""
        if self.keyword.num_docs > 0:
            return self.keyword.num_docs
        if self.numeric.num_docs > 0:
            return self.numeric.num_docs
        if self.date.num_docs > 0:
            return self.date.num_docs
        return 0

    def _apply_keyword(self, queries: list[KeywordQuery], mask: np.ndarray) -> np.ndarray:
        masker = self._create_masker(self.keyword.data)
        for query in queries:
            field_mask = masker.match_mask(query.field, query.value)
            mask = mask * field_mask
        return mask

    def _apply_numeric(self, queries: list[NumericExactQuery | NumericRangeQuery], mask: np.ndarray) -> np.ndarray:
        masker = self._create_masker(self.numeric.data)
        for query in queries:
            if isinstance(query, NumericRangeQuery):
                field_mask = masker.range_mask(query.field, query.conditions)
            else:
                field_mask = masker.match_mask(query.field, query.value)
            mask = mask * field_mask
        return mask

    def _apply_date(self, queries: list[DateExactQuery | DateRangeQuery], mask: np.ndarray) -> np.ndarray:
        masker = self._create_masker(self.date.data)
        for query in queries:
            if isinstance(query, DateRangeQuery):
                field_mask = masker.range_mask(query.field, query.conditions)
            else:
                field_mask = masker.match_mask(query.field, query.value)
            mask = mask * field_mask
        return mask
