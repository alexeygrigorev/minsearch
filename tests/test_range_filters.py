import pytest
from datetime import date, datetime
from minsearch.minsearch import Index


@pytest.fixture
def sample_docs_with_numeric():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language. It's easy to learn.",
            "price": 100,
            "rating": 4.5,
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI. It uses algorithms.",
            "price": 200,
            "rating": 3.8,
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality. Use pytest for Python.",
            "price": 150,
            "rating": 4.2,
        },
        {
            "question": "What is data science?",
            "text": "Data science involves statistics and programming.",
            "price": 50,
            "rating": 4.8,
        },
    ]


@pytest.fixture
def sample_docs_with_dates():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "created_at": date(2024, 1, 15),
        },
        {
            "question": "What is machine learning in Python?",
            "text": "Machine learning is a subset of AI.",
            "created_at": date(2024, 2, 20),
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality with Python.",
            "created_at": date(2024, 3, 10),
        },
        {
            "question": "What is data science?",
            "text": "Data science involves statistics.",
            "created_at": date(2024, 1, 5),
        },
    ]


@pytest.fixture
def sample_docs_with_datetime():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "created_at": datetime(2024, 1, 15, 10, 30),
        },
        {
            "question": "What is machine learning in Python?",
            "text": "Machine learning is a subset of AI.",
            "created_at": datetime(2024, 2, 20, 14, 45),
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality with Python.",
            "created_at": datetime(2024, 3, 10, 8, 0),
        },
    ]


def test_numeric_field_initialization():
    """Test that index initializes correctly with numeric fields."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price", "rating"]
    )
    assert index.text_fields == ["question", "text"]
    assert index.numeric_fields == ["price", "rating"]


def test_numeric_field_fit(sample_docs_with_numeric):
    """Test that fit works correctly with numeric fields."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price", "rating"]
    )
    index.fit(sample_docs_with_numeric)
    assert index.numeric_df is not None
    assert len(index.numeric_df) == 4
    assert list(index.numeric_df["price"]) == [100, 200, 150, 50]


def test_numeric_filter_greater_than_or_equal(sample_docs_with_numeric):
    """Test numeric filter with >= operator."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(sample_docs_with_numeric)

    # price >= 100
    results = index.search("python", filter_dict={"price": [('>=', 100)]})
    assert len(results) == 2
    assert all(doc["price"] >= 100 for doc in results)


def test_numeric_filter_less_than(sample_docs_with_numeric):
    """Test numeric filter with < operator."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(sample_docs_with_numeric)

    # price < 150
    results = index.search("python", filter_dict={"price": [('<', 150)]})
    assert len(results) == 1
    assert results[0]["price"] == 100


def test_numeric_filter_range(sample_docs_with_numeric):
    """Test numeric filter with range (AND of two conditions)."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(sample_docs_with_numeric)

    # 100 < price < 200
    results = index.search("python", filter_dict={"price": [('>', 100), ('<', 200)]})
    assert len(results) == 1
    assert results[0]["price"] == 150


def test_numeric_filter_greater_than_or_equal_and_less_than_or_equal(sample_docs_with_numeric):
    """Test numeric filter with >= and <= operators."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(sample_docs_with_numeric)

    # 100 <= price <= 200
    results = index.search("python", filter_dict={"price": [('>=', 100), ('<=', 200)]})
    assert len(results) == 2
    assert all(100 <= doc["price"] <= 200 for doc in results)


def test_numeric_filter_equals(sample_docs_with_numeric):
    """Test numeric filter with == operator."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(sample_docs_with_numeric)

    results = index.search("python", filter_dict={"price": [('==', 150)]})
    assert len(results) == 1
    assert results[0]["price"] == 150


def test_numeric_filter_not_equals(sample_docs_with_numeric):
    """Test numeric filter with != operator."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(sample_docs_with_numeric)

    results = index.search("python", filter_dict={"price": [('!=', 100)]})
    assert len(results) == 1
    assert all(doc["price"] != 100 for doc in results)


def test_numeric_filter_with_float_values(sample_docs_with_numeric):
    """Test numeric filter with float values."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["rating"]
    )
    index.fit(sample_docs_with_numeric)

    # rating >= 4.0
    results = index.search("python", filter_dict={"rating": [('>=', 4.0)]})
    assert len(results) == 2
    assert all(doc["rating"] >= 4.0 for doc in results)


def test_date_field_initialization():
    """Test that index initializes correctly with date fields."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        date_fields=["created_at"]
    )
    assert index.date_fields == ["created_at"]


def test_date_field_fit(sample_docs_with_dates):
    """Test that fit works correctly with date fields."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        date_fields=["created_at"]
    )
    index.fit(sample_docs_with_dates)
    assert index.date_df is not None
    assert len(index.date_df) == 4


def test_date_filter_greater_than_or_equal(sample_docs_with_dates):
    """Test date filter with >= operator."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        date_fields=["created_at"]
    )
    index.fit(sample_docs_with_dates)

    # created_at >= date(2024, 2, 1)
    results = index.search("python", filter_dict={"created_at": [('>=', date(2024, 2, 1))]})
    assert len(results) == 2
    assert all(doc["created_at"] >= date(2024, 2, 1) for doc in results)


def test_date_filter_range(sample_docs_with_dates):
    """Test date filter with range."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        date_fields=["created_at"]
    )
    index.fit(sample_docs_with_dates)

    # date(2024, 1, 10) <= created_at < date(2024, 3, 1)
    results = index.search("python", filter_dict={
        "created_at": [('>=', date(2024, 1, 10)), ('<', date(2024, 3, 1))]
    })
    assert len(results) == 2
    assert all(date(2024, 1, 10) <= doc["created_at"] < date(2024, 3, 1) for doc in results)


def test_datetime_filter(sample_docs_with_datetime):
    """Test datetime filter with datetime objects."""
    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        date_fields=["created_at"]
    )
    index.fit(sample_docs_with_datetime)

    # created_at >= datetime(2024, 2, 1, 0, 0)
    results = index.search("python", filter_dict={
        "created_at": [('>=', datetime(2024, 2, 1, 0, 0))]
    })
    assert len(results) == 2


def test_combined_filters(sample_docs_with_numeric):
    """Test combining keyword and numeric filters."""
    docs = [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "price": 100,
            "category": "programming",
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI.",
            "price": 200,
            "category": "ai",
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality with Python.",
            "price": 150,
            "category": "programming",
        },
    ]

    index = Index(
        text_fields=["question", "text"],
        keyword_fields=["category"],
        numeric_fields=["price"]
    )
    index.fit(docs)

    # category == "programming" AND price >= 100
    results = index.search("python", filter_dict={
        "category": "programming",
        "price": [('>=', 100)]
    })
    assert len(results) == 2
    assert all(doc["category"] == "programming" and doc["price"] >= 100 for doc in results)


def test_numeric_and_date_filters():
    """Test combining numeric and date filters."""
    docs = [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "price": 100,
            "created_at": date(2024, 1, 15),
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI.",
            "price": 200,
            "created_at": date(2024, 2, 20),
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality with Python.",
            "price": 150,
            "created_at": date(2024, 1, 20),
        },
    ]

    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"],
        date_fields=["created_at"]
    )
    index.fit(docs)

    # price >= 100 AND created_at >= date(2024, 1, 20)
    # Only the "How to write tests?" doc matches both filters and contains "python"
    results = index.search("python", filter_dict={
        "price": [('>=', 100)],
        "created_at": [('>=', date(2024, 1, 20))]
    })
    assert len(results) == 1
    assert results[0]["question"] == "How to write tests?"


def test_empty_numeric_field_values():
    """Test behavior when some documents are missing numeric field values."""
    docs = [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "price": 100,
        },
        {
            "question": "What is machine learning with Python?",
            "text": "Machine learning is a subset of AI.",
            # price missing
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality with Python.",
            "price": 150,
        },
    ]

    index = Index(
        text_fields=["question", "text"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(docs)

    # price >= 100 - should only return docs with explicit price >= 100
    results = index.search("python", filter_dict={"price": [('>=', 100)]})
    assert len(results) == 2
    assert all(doc.get("price", 0) >= 100 for doc in results)


def test_numeric_filter_no_matches(sample_docs_with_numeric):
    """Test numeric filter that matches no documents."""
    index = Index(
        text_fields=["question"],
        keyword_fields=[],
        numeric_fields=["price"]
    )
    index.fit(sample_docs_with_numeric)

    # price > 500 (no matches)
    results = index.search("python", filter_dict={"price": [('>', 500)]})
    assert len(results) == 0


def test_all_operators():
    """Test all supported operators."""
    docs = [
        {"question": "Test", "text": "Test", "value": 10},
        {"question": "Test", "text": "Test", "value": 20},
        {"question": "Test", "text": "Test", "value": 30},
    ]

    index = Index(
        text_fields=["question"],
        keyword_fields=[],
        numeric_fields=["value"]
    )
    index.fit(docs)

    # Test >=
    results = index.search("test", filter_dict={"value": [('>=', 20)]})
    assert len(results) == 2

    # Test >
    results = index.search("test", filter_dict={"value": [('>', 20)]})
    assert len(results) == 1

    # Test <=
    results = index.search("test", filter_dict={"value": [('<=', 20)]})
    assert len(results) == 2

    # Test <
    results = index.search("test", filter_dict={"value": [('<', 20)]})
    assert len(results) == 1

    # Test ==
    results = index.search("test", filter_dict={"value": [('==', 20)]})
    assert len(results) == 1

    # Test !=
    results = index.search("test", filter_dict={"value": [('!=', 20)]})
    assert len(results) == 2


def test_multiple_numeric_fields():
    """Test filtering on multiple numeric fields simultaneously."""
    docs = [
        {"question": "Test", "text": "Test", "price": 100, "rating": 4.5},
        {"question": "Test", "text": "Test", "price": 200, "rating": 3.8},
        {"question": "Test", "text": "Test", "price": 150, "rating": 4.2},
        {"question": "Test", "text": "Test", "price": 50, "rating": 4.8},
    ]

    index = Index(
        text_fields=["question"],
        keyword_fields=[],
        numeric_fields=["price", "rating"]
    )
    index.fit(docs)

    # price >= 100 AND rating >= 4.0
    results = index.search("test", filter_dict={
        "price": [('>=', 100)],
        "rating": [('>=', 4.0)]
    })
    assert len(results) == 2
    assert all(doc["price"] >= 100 and doc["rating"] >= 4.0 for doc in results)


def test_numeric_exact_match_equality():
    """Test numeric field exact match with simple value (not list)."""
    docs = [
        {"question": "Test", "text": "Test", "price": 100},
        {"question": "Test", "text": "Test", "price": 200},
        {"question": "Test", "text": "Test", "price": 150},
    ]

    index = Index(text_fields=["question"], numeric_fields=["price"])
    index.fit(docs)

    # Exact match: price == 150
    results = index.search("test", filter_dict={"price": 150})
    assert len(results) == 1
    assert results[0]["price"] == 150


def test_date_exact_match_equality():
    """Test date field exact match with simple value (not list)."""
    docs = [
        {"question": "Test", "text": "Test", "created_at": date(2024, 1, 15)},
        {"question": "Test", "text": "Test", "created_at": date(2024, 2, 20)},
        {"question": "Test", "text": "Test", "created_at": date(2024, 3, 10)},
    ]

    index = Index(text_fields=["question"], date_fields=["created_at"])
    index.fit(docs)

    # Exact match: created_at == date(2024, 2, 20)
    results = index.search("test", filter_dict={"created_at": date(2024, 2, 20)})
    assert len(results) == 1
    assert results[0]["created_at"] == date(2024, 2, 20)


def test_numeric_none_values():
    """Test numeric field with None values."""
    docs = [
        {"question": "Test doc one", "text": "Python test", "price": 100},
        {"question": "Test doc two", "text": "Python test", "price": None},
        {"question": "Test doc three", "text": "Python test", "price": 200},
    ]

    index = Index(text_fields=["question", "text"], numeric_fields=["price"])
    index.fit(docs)

    # price == None should find only docs with None price
    results = index.search("python", filter_dict={"price": None})
    assert len(results) == 1
    assert results[0]["price"] is None


def test_date_none_values():
    """Test date field with None values."""
    docs = [
        {"question": "Test doc one", "text": "Python test", "created_at": date(2024, 1, 15)},
        {"question": "Test doc two", "text": "Python test", "created_at": None},
        {"question": "Test doc three", "text": "Python test", "created_at": date(2024, 2, 20)},
    ]

    index = Index(text_fields=["question", "text"], date_fields=["created_at"])
    index.fit(docs)

    # created_at == None should find only docs with None date
    results = index.search("python", filter_dict={"created_at": None})
    assert len(results) == 1
    assert results[0]["created_at"] is None


def test_combined_equality_and_range_filters():
    """Test mixing equality (simple value) and range (list) filters."""
    docs = [
        {"question": "Test", "text": "Test", "price": 100, "category": "A"},
        {"question": "Test", "text": "Test", "price": 200, "category": "B"},
        {"question": "Test", "text": "Test", "price": 150, "category": "A"},
        {"question": "Test", "text": "Test", "price": 50, "category": "A"},
    ]

    index = Index(
        text_fields=["question"],
        keyword_fields=["category"],
        numeric_fields=["price"]
    )
    index.fit(docs)

    # category == "A" (equality) AND price >= 100 (range)
    results = index.search("test", filter_dict={
        "category": "A",
        "price": [('>=', 100)]
    })
    assert len(results) == 2
    assert all(doc["category"] == "A" and doc["price"] >= 100 for doc in results)


def test_zero_values_in_numeric_fields():
    """Test that zero values are handled correctly in numeric fields."""
    docs = [
        {"question": "Test doc one", "text": "Test content", "value": 0},
        {"question": "Test doc two", "text": "Test content", "value": 10},
        {"question": "Test doc three", "text": "Test content", "value": -5},
    ]

    index = Index(text_fields=["question", "text"], numeric_fields=["value"])
    index.fit(docs)

    # value == 0
    results = index.search("test", filter_dict={"value": 0})
    assert len(results) == 1
    assert results[0]["value"] == 0

    # value >= 0
    results = index.search("test", filter_dict={"value": [('>=', 0)]})
    assert len(results) == 2
    assert all(doc["value"] >= 0 for doc in results)
