"""
Query objects for validated filters.
"""
from dataclasses import dataclass
from enum import Enum
import operator as op
import pandas as pd
from datetime import date, datetime


class Operator(Enum):
    """Comparison operators for range queries."""

    def __init__(self, symbol: str, func):
        """Initialize operator with symbol and comparison function."""
        self.symbol = symbol
        self.func = func

    GREATER_EQUAL = ('>=', op.ge)
    GREATER = ('>', op.gt)
    LESS_EQUAL = ('<=', op.le)
    LESS = ('<', op.lt)
    EQUAL = ('==', op.eq)
    NOT_EQUAL = ('!=', op.ne)

    def __str__(self) -> str:
        """Return the symbol representation."""
        return self.symbol

    @classmethod
    def from_symbol(cls, symbol: str) -> 'Operator':
        """Get operator from its string symbol."""
        for op in cls:
            if op.symbol == symbol:
                return op
        raise ValueError(f"Unknown operator symbol: '{symbol}'")


@dataclass
class Condition:
    """A single condition in a range query."""
    operator: Operator
    value: int | float | date | datetime | pd.Timestamp


@dataclass
class KeywordQuery:
    """Exact match query for keyword fields."""
    field: str
    value: str | None


@dataclass
class NumericExactQuery:
    """Exact match query for numeric fields."""
    field: str
    value: int | float | None


@dataclass
class NumericRangeQuery:
    """Range query for numeric fields."""
    field: str
    conditions: list[Condition]


@dataclass
class DateExactQuery:
    """Exact match query for date fields."""
    field: str
    value: date | datetime | pd.Timestamp | None


@dataclass
class DateRangeQuery:
    """Range query for date fields."""
    field: str
    conditions: list[Condition]


@dataclass
class ValidatedFilter:
    """
    Structured representation of a validated filter.

    Created by Validator.validate() from a raw filter dictionary.
    Contains categorized query objects (KeywordQuery, NumericExactQuery, NumericRangeQuery,
    DateExactQuery, DateRangeQuery) for applying filters.
    """
    keyword_queries: list[KeywordQuery]
    numeric_queries: list[NumericExactQuery | NumericRangeQuery]
    date_queries: list[DateExactQuery | DateRangeQuery]
