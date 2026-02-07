"""
Query objects for validated filters.
"""
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import date, datetime


class Operator(Enum):
    """Comparison operators for range queries."""

    def __init__(self, symbol: str, func):
        """Initialize operator with symbol and comparison function."""
        self.symbol = symbol
        self.func = func

    GREATER_EQUAL = ('>=', lambda a, b: a >= b)
    GREATER = ('>', lambda a, b: a > b)
    LESS_EQUAL = ('<=', lambda a, b: a <= b)
    LESS = ('<', lambda a, b: a < b)
    EQUAL = ('==', lambda a, b: a == b)
    NOT_EQUAL = ('!=', lambda a, b: a != b)

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

    Created by Filter.validate() and consumed by Filter._apply_validated().
    """
    keyword_queries: list[KeywordQuery]
    numeric_queries: list[NumericExactQuery | NumericRangeQuery]
    date_queries: list[DateExactQuery | DateRangeQuery]
