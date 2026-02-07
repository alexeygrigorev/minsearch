"""
Tests for the filters package.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime

from minsearch.filters import (
    Operator,
    Condition,
    KeywordQuery,
    NumericExactQuery,
    NumericRangeQuery,
    DateExactQuery,
    DateRangeQuery,
    ValidatedFilter,
    Validator,
    FilterValidationError,
    PandasMasker,
    DictMasker,
    Filter,
    FieldData,
)


class TestOperator:
    def test_from_symbol(self):
        assert Operator.from_symbol('>=') == Operator.GREATER_EQUAL
        assert Operator.from_symbol('>') == Operator.GREATER
        assert Operator.from_symbol('<=') == Operator.LESS_EQUAL
        assert Operator.from_symbol('<') == Operator.LESS
        assert Operator.from_symbol('==') == Operator.EQUAL
        assert Operator.from_symbol('!=') == Operator.NOT_EQUAL

    def test_from_symbol_invalid(self):
        with pytest.raises(ValueError):
            Operator.from_symbol('invalid')

    def test_symbol(self):
        assert str(Operator.GREATER_EQUAL) == '>='
        assert str(Operator.LESS) == '<'

    def test_func(self):
        assert Operator.GREATER_EQUAL.func(5, 3) is True
        assert Operator.GREATER_EQUAL.func(3, 5) is False
        assert Operator.LESS.func(3, 5) is True
        assert Operator.EQUAL.func(5, 5) is True
        assert Operator.NOT_EQUAL.func(5, 3) is True


class TestCondition:
    def test_condition_creation(self):
        cond = Condition(Operator.GREATER_EQUAL, 10)
        assert cond.operator == Operator.GREATER_EQUAL
        assert cond.value == 10


class TestQueryObjects:
    def test_keyword_query(self):
        query = KeywordQuery(field='category', value='tech')
        assert query.field == 'category'
        assert query.value == 'tech'

    def test_numeric_exact_query(self):
        query = NumericExactQuery(field='price', value=100)
        assert query.field == 'price'
        assert query.value == 100

    def test_numeric_range_query(self):
        conditions = [Condition(Operator.GREATER_EQUAL, 10), Condition(Operator.LESS, 100)]
        query = NumericRangeQuery(field='price', conditions=conditions)
        assert query.field == 'price'
        assert len(query.conditions) == 2

    def test_date_exact_query(self):
        query = DateExactQuery(field='created', value=date(2024, 1, 1))
        assert query.field == 'created'
        assert query.value == date(2024, 1, 1)

    def test_date_range_query(self):
        conditions = [
            Condition(Operator.GREATER_EQUAL, date(2024, 1, 1)),
            Condition(Operator.LESS, date(2024, 12, 31))
        ]
        query = DateRangeQuery(field='created', conditions=conditions)
        assert query.field == 'created'
        assert len(query.conditions) == 2


class TestValidatedFilter:
    def test_empty_filter(self):
        vf = ValidatedFilter(keyword_queries=[], numeric_queries=[], date_queries=[])
        assert len(vf.keyword_queries) == 0
        assert len(vf.numeric_queries) == 0
        assert len(vf.date_queries) == 0


class TestValidator:
    def setup_method(self):
        self.validator = Validator(
            keyword_fields=['category', 'status'],
            numeric_fields=['price', 'quantity'],
            date_fields=['created', 'updated']
        )

    def test_validate_keyword_filter(self):
        result = self.validator.validate({'category': 'tech'})
        assert len(result.keyword_queries) == 1
        assert result.keyword_queries[0].field == 'category'
        assert result.keyword_queries[0].value == 'tech'

    def test_validate_numeric_exact_filter(self):
        result = self.validator.validate({'price': 100})
        assert len(result.numeric_queries) == 1
        assert isinstance(result.numeric_queries[0], NumericExactQuery)
        assert result.numeric_queries[0].value == 100

    def test_validate_numeric_range_filter(self):
        result = self.validator.validate({'price': [('>=', 10), ('<', 100)]})
        assert len(result.numeric_queries) == 1
        assert isinstance(result.numeric_queries[0], NumericRangeQuery)
        assert len(result.numeric_queries[0].conditions) == 2

    def test_validate_date_exact_filter(self):
        result = self.validator.validate({'created': date(2024, 1, 1)})
        assert len(result.date_queries) == 1
        assert isinstance(result.date_queries[0], DateExactQuery)
        assert isinstance(result.date_queries[0].value, pd.Timestamp)
        assert result.date_queries[0].value == pd.Timestamp(date(2024, 1, 1))

    def test_validate_date_range_filter(self):
        result = self.validator.validate({
            'created': [('>=', date(2024, 1, 1)), ('<', date(2025, 1, 1))]
        })
        assert len(result.date_queries) == 1
        assert isinstance(result.date_queries[0], DateRangeQuery)
        assert len(result.date_queries[0].conditions) == 2

    def test_validate_multiple_filters(self):
        result = self.validator.validate({
            'category': 'tech',
            'price': 100,
            'created': date(2024, 1, 1)
        })
        assert len(result.keyword_queries) == 1
        assert len(result.numeric_queries) == 1
        assert len(result.date_queries) == 1

    def test_validate_none_filter(self):
        result = self.validator.validate({'price': None})
        assert len(result.numeric_queries) == 1
        assert result.numeric_queries[0].value is None

    def test_validate_invalid_field(self):
        with pytest.raises(FilterValidationError, match="Unknown filter field 'unknown'"):
            self.validator.validate({'unknown': 'value'})

    def test_validate_invalid_dict(self):
        with pytest.raises(FilterValidationError, match="filter_dict must be a dictionary"):
            self.validator.validate("not a dict")

    def test_validate_invalid_operator(self):
        with pytest.raises(FilterValidationError, match="Unknown operator"):
            self.validator.validate({'price': [('invalid', 10)]})

    def test_validate_invalid_date_range_format(self):
        with pytest.raises(FilterValidationError, match="must be a list of tuples"):
            self.validator.validate({'created': ['not a tuple']})

    def test_validate_date_range_with_string(self):
        with pytest.raises(FilterValidationError, match="requires date/datetime objects"):
            self.validator.validate({'created': [('>=', '2024-01-01')]})


class TestPandasMasker:
    def setup_method(self):
        self.data = pd.DataFrame({
            'category': ['tech', 'tech', 'health'],
            'status': ['active', None, 'active'],
            'price': [100, 200, 50],
            'quantity': [10, None, 5],
            'created': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'updated': [datetime(2024, 1, 15), None, datetime(2024, 3, 15)]
        })
        self.masker = PandasMasker(data=self.data, num_docs=3)

    def test_match_mask_keyword(self):
        mask = self.masker.match_mask('category', 'tech')
        np.testing.assert_array_equal(mask, [1, 1, 0])

    def test_match_mask_numeric(self):
        mask = self.masker.match_mask('price', 100)
        np.testing.assert_array_equal(mask, [1, 0, 0])

    def test_match_mask_none(self):
        mask = self.masker.match_mask('status', None)
        np.testing.assert_array_equal(mask, [0, 1, 0])

    def test_range_mask_numeric(self):
        conditions = [Condition(Operator.GREATER_EQUAL, 100), Condition(Operator.LESS, 200)]
        mask = self.masker.range_mask('price', conditions)
        np.testing.assert_array_equal(mask, [1, 0, 0])

    def test_range_mask_date(self):
        start = date(2024, 2, 1)
        end = date(2024, 3, 1)
        conditions = [Condition(Operator.GREATER_EQUAL, start), Condition(Operator.LESS_EQUAL, end)]
        mask = self.masker.range_mask('created', conditions)
        np.testing.assert_array_equal(mask, [0, 1, 1])


class TestDictMasker:
    def setup_method(self):
        self.data = {
            'category': ['tech', 'tech', 'health'],
            'status': ['active', None, 'active'],
            'price': [100, 200, 50],
            'quantity': [10, None, 5],
            'created': [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
            'updated': [datetime(2024, 1, 15), None, datetime(2024, 3, 15)]
        }
        self.masker = DictMasker(data=self.data, num_docs=3)

    def test_match_mask_keyword(self):
        mask = self.masker.match_mask('category', 'tech')
        np.testing.assert_array_equal(mask, [1, 1, 0])

    def test_match_mask_numeric(self):
        mask = self.masker.match_mask('price', 100)
        np.testing.assert_array_equal(mask, [1, 0, 0])

    def test_match_mask_none(self):
        mask = self.masker.match_mask('status', None)
        np.testing.assert_array_equal(mask, [0, 1, 0])

    def test_range_mask_numeric(self):
        conditions = [Condition(Operator.GREATER_EQUAL, 100), Condition(Operator.LESS, 200)]
        mask = self.masker.range_mask('price', conditions)
        np.testing.assert_array_equal(mask, [1, 0, 0])

    def test_range_mask_date(self):
        start = date(2024, 2, 1)
        end = date(2024, 3, 1)
        conditions = [Condition(Operator.GREATER_EQUAL, start), Condition(Operator.LESS_EQUAL, end)]
        mask = self.masker.range_mask('created', conditions)
        np.testing.assert_array_equal(mask, [0, 1, 1])


class TestFilter:
    def setup_method(self):
        self.keyword_df = pd.DataFrame({
            'category': ['tech', 'tech', 'health'],
            'status': ['active', None, 'active']
        })
        self.numeric_df = pd.DataFrame({
            'price': [100, 200, 50],
            'quantity': [10, None, 5]
        })
        self.date_df = pd.DataFrame({
            'created': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'updated': [datetime(2024, 1, 15), None, datetime(2024, 3, 15)]
        })
        self.filter = Filter(
            keyword=FieldData(fields=['category', 'status'], data=self.keyword_df),
            numeric=FieldData(fields=['price', 'quantity'], data=self.numeric_df),
            date=FieldData(fields=['created', 'updated'], data=self.date_df),
        )

    def test_apply_empty_filter(self):
        mask = self.filter.apply({})
        np.testing.assert_array_equal(mask, [1, 1, 1])

    def test_apply_keyword_filter(self):
        mask = self.filter.apply({'category': 'tech'})
        np.testing.assert_array_equal(mask, [1, 1, 0])

    def test_apply_numeric_exact_filter(self):
        mask = self.filter.apply({'price': 100})
        np.testing.assert_array_equal(mask, [1, 0, 0])

    def test_apply_numeric_range_filter(self):
        mask = self.filter.apply({'price': [('>=', 50), ('<', 150)]})
        np.testing.assert_array_equal(mask, [1, 0, 1])

    def test_apply_date_exact_filter(self):
        mask = self.filter.apply({'created': date(2024, 2, 1)})
        np.testing.assert_array_equal(mask, [0, 1, 0])

    def test_apply_none_filter(self):
        mask = self.filter.apply({'status': None})
        np.testing.assert_array_equal(mask, [0, 1, 0])

    def test_apply_multiple_filters(self):
        mask = self.filter.apply({
            'category': 'tech',
            'price': [('>=', 50), ('<', 150)]
        })
        np.testing.assert_array_equal(mask, [1, 0, 0])

    def test_refresh(self):
        new_numeric_df = pd.DataFrame({'price': [150, 250, 75], 'quantity': [15, 25, 35]})
        self.filter.refresh(numeric_data=new_numeric_df, num_docs=3)
        mask = self.filter.apply({'price': 150})
        np.testing.assert_array_equal(mask, [1, 0, 0])


class TestFilterWithDictData:
    def setup_method(self):
        self.keyword_data = {
            'category': ['tech', 'tech', 'health'],
            'status': ['active', None, 'active']
        }
        self.numeric_data = {
            'price': [100, 200, 50],
            'quantity': [10, None, 5]
        }
        self.date_data = {
            'created': [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
            'updated': [datetime(2024, 1, 15), None, datetime(2024, 3, 15)]
        }
        self.filter = Filter(
            keyword=FieldData(fields=['category', 'status'], data=self.keyword_data),
            numeric=FieldData(fields=['price', 'quantity'], data=self.numeric_data),
            date=FieldData(fields=['created', 'updated'], data=self.date_data),
        )

    def test_apply_keyword_filter_dict(self):
        mask = self.filter.apply({'category': 'tech'})
        np.testing.assert_array_equal(mask, [1, 1, 0])

    def test_apply_numeric_range_filter_dict(self):
        mask = self.filter.apply({'price': [('>=', 50), ('<', 150)]})
        np.testing.assert_array_equal(mask, [1, 0, 1])

    def test_apply_date_range_filter_dict(self):
        start = date(2024, 2, 1)
        end = date(2024, 3, 1)
        mask = self.filter.apply({'created': [('>=', start), ('<=', end)]})
        np.testing.assert_array_equal(mask, [0, 1, 1])
