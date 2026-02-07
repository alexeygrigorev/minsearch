"""
Validator for filter dictionaries.
"""
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


class FilterValidationError(ValueError):
    """Raised when filter validation fails."""


class Validator:
    """Validates filter dictionaries and creates query objects."""

    def __init__(
        self,
        keyword_fields: list[str],
        numeric_fields: list[str],
        date_fields: list[str],
    ):
        self.keyword_fields = keyword_fields
        self.numeric_fields = numeric_fields
        self.date_fields = date_fields

    def validate(self, filter_dict: dict) -> ValidatedFilter:
        """Validate and categorize filter dictionary into query objects."""
        if not isinstance(filter_dict, dict):
            raise FilterValidationError("filter_dict must be a dictionary")

        keyword_queries: list[KeywordQuery] = []
        numeric_queries: list[NumericExactQuery | NumericRangeQuery] = []
        date_queries: list[DateExactQuery | DateRangeQuery] = []

        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                keyword_queries.append(KeywordQuery(field=field, value=value))
            elif field in self.numeric_fields:
                self._validate_numeric(field, value)
                if value is None:
                    numeric_queries.append(NumericExactQuery(field=field, value=None))
                elif isinstance(value, list) and all(isinstance(v, tuple) and len(v) == 2 for v in value):
                    conditions = [
                        Condition(operator=self._parse_operator(op), value=val)
                        for op, val in value
                    ]
                    numeric_queries.append(NumericRangeQuery(field=field, conditions=conditions))
                else:
                    numeric_queries.append(NumericExactQuery(field=field, value=value))
            elif field in self.date_fields:
                self._validate_date(field, value)
                if value is None:
                    date_queries.append(DateExactQuery(field=field, value=None))
                elif isinstance(value, list) and all(isinstance(v, tuple) and len(v) == 2 for v in value):
                    conditions = [
                        Condition(operator=self._parse_operator(op), value=val)
                        for op, val in value
                    ]
                    date_queries.append(DateRangeQuery(field=field, conditions=conditions))
                else:
                    exact_value = value
                    if isinstance(value, (date, datetime)):
                        exact_value = pd.Timestamp(value)
                    date_queries.append(DateExactQuery(field=field, value=exact_value))
            else:
                raise FilterValidationError(
                    f"Unknown filter field '{field}'. "
                    f"Valid fields are: {self.keyword_fields + self.numeric_fields + self.date_fields}"
                )

        return ValidatedFilter(
            keyword_queries=keyword_queries,
            numeric_queries=numeric_queries,
            date_queries=date_queries,
        )

    def _parse_operator(self, op_str: str) -> Operator:
        """Parse operator string to Operator enum."""
        try:
            return Operator.from_symbol(op_str)
        except ValueError as e:
            raise FilterValidationError(
                f"{str(e)}. Valid operators are: {[op.symbol for op in Operator]}"
            )

    def _validate_date(self, field: str, value) -> None:
        """Validate date field filter values."""
        if value is None:
            return

        if isinstance(value, list):
            if not all(isinstance(v, tuple) and len(v) == 2 for v in value):
                raise FilterValidationError(
                    f"Date range filter for '{field}' must be a list of tuples"
                )
            for op, op_value in value:
                try:
                    Operator.from_symbol(op)
                except ValueError:
                    raise FilterValidationError(
                        f"Unknown operator '{op}' for date field '{field}'. "
                        f"Valid operators are: {[op.symbol for op in Operator]}"
                    )
                if op_value is not None and not isinstance(op_value, (date, datetime, pd.Timestamp)):
                    raise FilterValidationError(
                        f"Date field '{field}' requires date/datetime objects for range filters"
                    )
        elif not isinstance(value, (date, datetime, pd.Timestamp, str)):
            raise FilterValidationError(
                f"Date field '{field}' requires date/datetime objects"
            )

    def _validate_numeric(self, field: str, value) -> None:
        """Validate numeric field filter values."""
        if value is None:
            return

        if isinstance(value, list):
            if not all(isinstance(v, tuple) and len(v) == 2 for v in value):
                raise FilterValidationError(
                    f"Numeric range filter for '{field}' must be a list of tuples"
                )
            for op, op_value in value:
                try:
                    Operator.from_symbol(op)
                except ValueError:
                    raise FilterValidationError(
                        f"Unknown operator '{op}' for numeric field '{field}'. "
                        f"Valid operators are: {[op.symbol for op in Operator]}"
                    )
