import pytest
from datetime import date, datetime
from minsearch.minsearch import Index
from minsearch.append import AppendableIndex


@pytest.fixture
def sample_docs():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language. It's easy to learn.",
            "section": "Programming",
            "course": "python-basics",
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI. It uses algorithms.",
            "section": "AI",
            "course": "ml-basics",
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality. Use pytest for Python.",
            "section": "Testing",
            "course": "python-basics",
        },
    ]


@pytest.fixture
def text_fields():
    return ["question", "text", "section"]


@pytest.fixture
def keyword_fields():
    return ["course"]


def test_index_initialization(text_fields, keyword_fields):
    """Test that both index types initialize correctly."""
    index = Index(text_fields, keyword_fields)
    appendable_index = AppendableIndex(text_fields, keyword_fields)

    assert index.text_fields == text_fields
    assert index.keyword_fields == keyword_fields
    assert appendable_index.text_fields == text_fields
    assert appendable_index.keyword_fields == keyword_fields


def test_fit_and_search(text_fields, keyword_fields, sample_docs):
    """Test that both index types return the same results after fit."""
    index = Index(text_fields, keyword_fields)
    appendable_index = AppendableIndex(text_fields, keyword_fields)

    # Fit both indices
    index.fit(sample_docs)
    appendable_index.fit(sample_docs)

    # Test search with different queries
    queries = ["python", "machine learning", "testing", "nonexistent"]

    for query in queries:
        # Search with no filters
        index_results = index.search(query)
        appendable_results = appendable_index.search(query)
        assert len(index_results) == len(appendable_results)

        # Search with filters
        filter_dict = {"course": "python-basics"}
        index_results = index.search(query, filter_dict=filter_dict)
        appendable_results = appendable_index.search(query, filter_dict=filter_dict)
        assert len(index_results) == len(appendable_results)

        # Search with boosts
        boost_dict = {"question": 2, "text": 1}
        index_results = index.search(query, boost_dict=boost_dict)
        appendable_results = appendable_index.search(query, boost_dict=boost_dict)
        assert len(index_results) == len(appendable_results)


def test_appendable_index_append(text_fields, keyword_fields, sample_docs):
    """Test that appendable index works correctly with append."""
    appendable_index = AppendableIndex(text_fields, keyword_fields)

    # First fit with initial docs
    initial_docs = sample_docs[:2]
    appendable_index.fit(initial_docs)

    # Then append the last doc
    appendable_index.append(sample_docs[2])

    # Search should work the same as if we had fit all docs at once
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)

    # Test with different queries
    queries = ["python", "machine learning", "testing", "nonexistent"]

    for query in queries:
        index_results = index.search(query)
        appendable_results = appendable_index.search(query)
        assert len(index_results) == len(appendable_results)


def test_stop_words(text_fields, keyword_fields, sample_docs):
    """Test that stop words are properly removed."""
    # Create index with default stop words
    index = AppendableIndex(text_fields, keyword_fields)
    index.fit(sample_docs)

    # Search for a query containing stop words
    results = index.search("how to use the python")

    # Should still find relevant documents
    assert len(results) > 0
    assert any("python" in doc["text"].lower() for doc in results)


def test_empty_docs(text_fields, keyword_fields):
    """Test behavior with empty document list."""
    index = Index(text_fields, keyword_fields)
    appendable_index = AppendableIndex(text_fields, keyword_fields)

    # Fit with empty docs
    index.fit([])
    appendable_index.fit([])

    # Search should return empty list
    assert len(index.search("python")) == 0
    assert len(appendable_index.search("python")) == 0


def test_empty_fields(text_fields, keyword_fields, sample_docs):
    """Test behavior with empty field values."""
    # Add a doc with empty fields
    docs_with_empty = sample_docs + [
        {"question": "", "text": "", "section": "", "course": ""}
    ]

    index = Index(text_fields, keyword_fields)
    appendable_index = AppendableIndex(text_fields, keyword_fields)

    index.fit(docs_with_empty)
    appendable_index.fit(docs_with_empty)

    # Search should still work
    results = index.search("python")
    appendable_results = appendable_index.search("python")
    assert len(results) == len(appendable_results)


def test_boost_weights(text_fields, keyword_fields, sample_docs):
    """Test that boost weights affect search results."""
    index = Index(text_fields, keyword_fields)
    appendable_index = AppendableIndex(text_fields, keyword_fields)

    index.fit(sample_docs)
    appendable_index.fit(sample_docs)

    # Search with different boost weights
    boost_configs = [
        {"question": 1, "text": 1, "section": 1},
        {"question": 2, "text": 1, "section": 1},
        {"question": 1, "text": 2, "section": 1},
        {"question": 1, "text": 1, "section": 2},
    ]

    for boost_dict in boost_configs:
        index_results = index.search("python", boost_dict=boost_dict)
        appendable_results = appendable_index.search("python", boost_dict=boost_dict)
        assert len(index_results) == len(appendable_results)


def test_filter_combinations(text_fields, keyword_fields, sample_docs):
    """Test different combinations of filters."""
    index = Index(text_fields, keyword_fields)
    appendable_index = AppendableIndex(text_fields, keyword_fields)

    index.fit(sample_docs)
    appendable_index.fit(sample_docs)

    # Test different filter combinations
    filter_configs = [
        {"course": "python-basics"},
        {"course": "ml-basics"},
        {"course": "nonexistent"},
    ]

    for filter_dict in filter_configs:
        index_results = index.search("python", filter_dict=filter_dict)
        appendable_results = appendable_index.search("python", filter_dict=filter_dict)
        assert len(index_results) == len(appendable_results)


def test_basic_search():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = AppendableIndex(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    results = index.search("python")
    assert len(results) > 0
    assert results[0]["title"] == "Python Programming"


def test_filter_combinations():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = AppendableIndex(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    # Test single filter
    results = index.search("programming", filter_dict={"course": "CS101"})
    assert len(results) > 0
    assert results[0]["course"] == "CS101"

    # Test non-existent filter value (returns no results)
    results = index.search("programming", filter_dict={"course": "non_existent"})
    assert len(results) == 0

    # Test multiple filters (same filter, not multiple different fields)
    results = index.search(
        "programming", filter_dict={"course": "CS101"}
    )
    assert len(results) > 0


def test_boost_combinations():
    docs = [
        {"title": "Python Programming", "description": "Learn Python programming"},
        {"title": "Data Science", "description": "Python for data science"},
    ]
    index = AppendableIndex(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    # Test without boost
    results = index.search("python")
    assert len(results) > 0

    # Test with boost
    results = index.search("python", boost_dict={"title": 2.0})
    assert len(results) > 0


def test_empty_docs():
    index = AppendableIndex(text_fields=["title"], keyword_fields=["course"])
    index.fit([])

    results = index.search("python")
    assert len(results) == 0


def test_stop_words():
    docs = [
        {
            "title": "The Python Programming",
            "description": "Learn the Python programming",
        },
        {"title": "Data Science", "description": "Python for data science"},
    ]

    # Test with default stop words
    index = AppendableIndex(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    results = index.search("the python")
    assert len(results) > 0

    # Test with custom stop words
    index = AppendableIndex(
        text_fields=["title", "description"],
        keyword_fields=[],
        stop_words={"the", "a", "an"},
    )
    index.fit(docs)

    results = index.search("the python")
    assert len(results) > 0


def test_empty_query():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = AppendableIndex(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    results = index.search("")
    assert len(results) == 0


def test_special_characters():
    docs = [
        {"title": "Python-Programming", "description": "Learn Python (programming)"},
        {"title": "Data-Science", "description": "Python for data-science"},
    ]
    index = AppendableIndex(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    results = index.search("python-programming")
    assert len(results) > 0
    assert results[0]["title"] == "Python-Programming"


def test_append():
    # Test the append functionality
    index = AppendableIndex(text_fields=["title"], keyword_fields=["course"])

    # Add first document
    doc1 = {"title": "Python Programming", "course": "CS101"}
    index.append(doc1)

    # Search after first append
    results = index.search("python")
    assert len(results) == 1
    assert results[0]["title"] == "Python Programming"

    # Add second document
    doc2 = {"title": "Data Science", "course": "CS102"}
    index.append(doc2)

    # Search after second append
    results = index.search("python")
    assert len(results) == 1
    assert results[0]["title"] == "Python Programming"

    # Search for second document
    results = index.search("data")
    assert len(results) == 1
    assert results[0]["title"] == "Data Science"


def test_append_after_fit():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = AppendableIndex(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    # Append new document
    doc3 = {"title": "Machine Learning", "course": "CS103"}
    index.append(doc3)

    # Search for new document
    results = index.search("machine")
    assert len(results) == 1
    assert results[0]["title"] == "Machine Learning"

    # Search for existing documents
    results = index.search("python")
    assert len(results) == 1
    assert results[0]["title"] == "Python Programming"


def test_readme_example():
    # Test the exact example from README
    index = AppendableIndex(
        text_fields=["title", "description"], keyword_fields=["course"]
    )

    # Add documents one by one
    doc1 = {
        "title": "Python Programming",
        "description": "Learn Python programming",
        "course": "CS101",
    }
    index.append(doc1)

    doc2 = {
        "title": "Data Science",
        "description": "Python for data science",
        "course": "CS102",
    }
    index.append(doc2)

    doc3 = {
        "title": "Machine Learning",
        "description": "Introduction to ML",
        "course": "CS103",
    }
    index.append(doc3)

    # Test basic search
    results = index.search("python")
    assert len(results) > 0
    assert results[0]["title"] == "Python Programming"

    # Test with filters
    results = index.search("python", filter_dict={"course": "CS101"})
    assert len(results) > 0
    assert results[0]["course"] == "CS101"

    # Test with boost
    results = index.search("python", boost_dict={"title": 2.0})
    assert len(results) > 0

    # Test with single filter
    results = index.search(
        "python", filter_dict={"course": "CS101"}
    )
    assert len(results) > 0


def test_tokenizer():
    # Test the custom tokenizer
    docs = [
        {
            "title": "The Python Programming",
            "description": "Learn the Python programming",
        },
        {"title": "Data Science", "description": "Python for data science"},
    ]

    # Test with default tokenizer
    index = AppendableIndex(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    results = index.search("the python")
    assert len(results) > 0

    # Test with custom tokenizer pattern
    index = AppendableIndex(
        text_fields=["title", "description"],
        keyword_fields=[],
        stop_words={"the", "a", "an"},
    )
    index.fit(docs)

    results = index.search("the python")
    assert len(results) > 0


def test_boost_affects_ranking():
    """Test that boost parameters affect the ranking of search results."""
    docs = [
        {
            "title": "Introduction to Programming",
            "description": "Python is a popular programming language. Python is used in many applications.",
            "course": "CS101",
        },
        {
            "title": "Python for Beginners",
            "description": "Learn the basics of programming",
            "course": "CS102",
        },
        {
            "title": "Advanced Topics",
            "description": "Python is essential for data science. Python is used in machine learning.",
            "course": "CS103",
        },
    ]

    index = AppendableIndex(
        text_fields=["title", "description"], keyword_fields=["course"]
    )
    index.fit(docs)

    # Search without boost
    results_no_boost = index.search("python")
    assert len(results_no_boost) > 0

    # Search with title boost (high value to ensure it affects ranking)
    results_title_boost = index.search("python", boost_dict={"title": 10.0})
    assert len(results_title_boost) > 0

    # Search with description boost (high value to ensure it affects ranking)
    results_desc_boost = index.search("python", boost_dict={"description": 10.0})
    assert len(results_desc_boost) > 0

    # Verify that boosting affects ranking
    # When title is boosted, documents with "Python" in title should rank higher
    title_boosted_first = results_title_boost[0]
    assert "Python" in title_boosted_first["title"], (
        "Title boost should rank documents with Python in title higher"
    )

    # When description is boosted, documents with more mentions of "Python" in description should rank higher
    desc_boosted_first = results_desc_boost[0]
    assert "Python" in desc_boosted_first["description"], (
        "Description boost should rank documents with Python in description higher"
    )

    # The rankings should be different between no boost and boosted searches
    # Check that the first result is different between searches
    assert (
        results_no_boost[0] != results_title_boost[0]
        or results_no_boost[0] != results_desc_boost[0]
    ), "Boosting should change the ranking"

    # Additional checks for specific ranking changes
    # With title boost, documents with Python in title should be ranked higher
    title_boosted_titles = [doc["title"] for doc in results_title_boost]
    assert "Python" in title_boosted_titles[0], (
        "Title boost should rank documents with Python in title higher"
    )

    # With description boost, documents with more Python mentions in description should be ranked higher
    desc_boosted_descriptions = [doc["description"] for doc in results_desc_boost]
    assert "Python" in desc_boosted_descriptions[0], (
        "Description boost should rank documents with Python in description higher"
    )

    # Verify that the document with most Python mentions in description is ranked highest with description boost
    python_mentions = [
        doc["description"].lower().count("python") for doc in results_desc_boost
    ]
    assert python_mentions[0] == max(python_mentions), (
        "Description boost should rank documents with more Python mentions higher"
    )


def test_output_ids(text_fields, keyword_fields, sample_docs):
    """Test that output_ids parameter works correctly."""
    index = AppendableIndex(text_fields, keyword_fields)
    index.fit(sample_docs)

    # Test without output_ids
    results = index.search("python")
    assert len(results) > 0
    assert isinstance(results[0], dict)
    assert "_id" not in results[0]

    # Test with output_ids
    results_with_ids = index.search("python", output_ids=True)
    assert len(results_with_ids) > 0
    assert isinstance(results_with_ids[0], dict)
    assert "_id" in results_with_ids[0]
    assert isinstance(results_with_ids[0]["_id"], int)

    # Test that IDs match document positions
    for doc in results_with_ids:
        assert doc == {**sample_docs[doc["_id"]], "_id": doc["_id"]}


def test_appended_document_ranking_matches_index():
    """
    Test that documents appended to AppendableIndex rank similarly to
    when the same document is included in a regular Index from the start.

    This is a regression test for an issue where appended documents didn't
    appear in search results because of incorrect L2 normalization in the
    TF-IDF calculation.
    """
    # Create a corpus with several documents
    initial_docs = [
        {
            "question": "What is Docker?",
            "text": "Docker is a container platform.",
            "section": "Module 1: Docker and Terraform",
            "course": "de-zoomcamp",
        },
        {
            "question": "How to use Docker Compose?",
            "text": "Docker Compose helps manage multiple containers.",
            "section": "Module 1: Docker and Terraform",
            "course": "de-zoomcamp",
        },
        {
            "question": "What is PostgreSQL?",
            "text": "PostgreSQL is a relational database.",
            "section": "Module 1: Docker and Terraform",
            "course": "de-zoomcamp",
        },
        {
            "question": "How to connect to PostgreSQL?",
            "text": "Use psycopg2 to connect Python to PostgreSQL.",
            "section": "Module 1: Docker and Terraform",
            "course": "de-zoomcamp",
        },
    ]

    # A new document to be appended
    new_doc = {
        "question": "How do I do well in Docker during Module 1?",
        "text": "Learn Docker concepts and practice with containers.",
        "section": "user added",  # Different section than the corpus
        "course": "de-zoomcamp",
    }

    text_fields = ["question", "text", "section"]
    keyword_fields = ["course"]

    # Create AppendableIndex with fit + append
    appendable_index = AppendableIndex(text_fields, keyword_fields)
    appendable_index.fit(initial_docs)
    appendable_index.append(new_doc)

    # Create regular Index with all docs
    all_docs = initial_docs + [new_doc]
    regular_index = Index(text_fields, keyword_fields)
    regular_index.fit(all_docs)

    # Search for the exact question of the new document
    query = "How do I do well in Docker during Module 1?"

    regular_results = regular_index.search(query, num_results=10)
    appendable_results = appendable_index.search(query, num_results=10)

    # The new document should appear in both result sets
    regular_questions = [r["question"] for r in regular_results]
    appendable_questions = [r["question"] for r in appendable_results]

    # Check that the new document is found in both indices
    assert new_doc["question"] in regular_questions, (
        "New document should be found in regular Index results"
    )
    assert new_doc["question"] in appendable_questions, (
        "New document should be found in AppendableIndex results"
    )


def test_appended_document_normalized_correctly():
    """
    Test that documents with different numbers of tokens are normalized
    correctly, matching sklearn's TF-IDF behavior.

    Documents with more non-query tokens should have lower scores than
    documents with fewer non-query tokens (all else being equal).
    """
    # Two documents: one with only query terms, one with extra terms
    docs = [
        {"text": "docker module"},  # Only query terms
        {"text": "docker module extra terms here"},  # Query terms + extras
    ]

    index = AppendableIndex(text_fields=["text"], keyword_fields=[])
    index.fit(docs)

    # Search for "docker module"
    results = index.search("docker module", num_results=10)

    # The first document (fewer tokens) should rank higher than the second
    assert results[0]["text"] == "docker module", (
        "Document with only query terms should rank higher"
    )
    assert results[1]["text"] == "docker module extra terms here", (
        "Document with extra terms should rank lower"
    )


class TestAppendableIndexRangeFilters:
    """Tests for numeric and date range filters in AppendableIndex."""

    def test_numeric_filter_greater_than_or_equal(self):
        """Test numeric filter with >= operator."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "price": 100},
            {"question": "What is machine learning?", "text": "ML is AI.", "price": 200},
            {"question": "How to write tests?", "text": "Tests with Python.", "price": 150},
            {"question": "What is data science?", "text": "Data science.", "price": 50},
        ]

        index = AppendableIndex(text_fields=["question", "text"], numeric_fields=["price"])
        index.fit(docs)

        results = index.search("python", filter_dict={"price": [('>=', 100)]})
        assert all(doc["price"] >= 100 for doc in results)

    def test_numeric_filter_range(self):
        """Test numeric filter with range."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "price": 100},
            {"question": "What is machine learning?", "text": "ML is AI.", "price": 200},
            {"question": "How to write tests?", "text": "Tests with Python.", "price": 150},
            {"question": "What is data science?", "text": "Data science.", "price": 50},
        ]

        index = AppendableIndex(text_fields=["question", "text"], numeric_fields=["price"])
        index.fit(docs)

        results = index.search("python", filter_dict={"price": [('>', 100), ('<', 200)]})
        assert len(results) == 1
        assert results[0]["price"] == 150

    def test_date_filter_greater_than_or_equal(self):
        """Test date filter with >= operator."""
        from datetime import date

        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "created_at": date(2024, 1, 15)},
            {"question": "What is ML with Python?", "text": "ML is AI.", "created_at": date(2024, 2, 20)},
            {"question": "How to write tests?", "text": "Tests with Python.", "created_at": date(2024, 3, 10)},
            {"question": "What is data science?", "text": "Data science.", "created_at": date(2024, 1, 5)},
        ]

        index = AppendableIndex(text_fields=["question", "text"], date_fields=["created_at"])
        index.fit(docs)

        results = index.search("python", filter_dict={"created_at": [('>=', date(2024, 2, 1))]})
        assert all(doc["created_at"] >= date(2024, 2, 1) for doc in results)

    def test_combined_keyword_and_numeric_filters(self):
        """Test combining keyword and numeric filters."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "price": 100, "category": "A"},
            {"question": "What is machine learning?", "text": "ML is AI.", "price": 200, "category": "B"},
            {"question": "How to write tests in Python?", "text": "Tests with Python.", "price": 150, "category": "A"},
            {"question": "What is data science?", "text": "Data science.", "price": 50, "category": "B"},
        ]

        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["category"],
            numeric_fields=["price"]
        )
        index.fit(docs)

        results = index.search("python", filter_dict={
            "category": "A",
            "price": [('>=', 100)]
        })
        assert all(doc["category"] == "A" and doc["price"] >= 100 for doc in results)

    def test_append_with_numeric_field(self):
        """Test that appending preserves numeric field functionality."""
        index = AppendableIndex(text_fields=["question"], numeric_fields=["price"])

        index.append({"question": "How do I use Python?", "price": 100})
        index.append({"question": "What is machine learning?", "price": 200})
        index.append({"question": "How to write tests in Python?", "price": 150})

        results = index.search("python", filter_dict={"price": [('>=', 150)]})
        assert all(doc["price"] >= 150 for doc in results)

    def test_append_with_date_field(self):
        """Test that appending preserves date field functionality."""
        from datetime import date

        index = AppendableIndex(text_fields=["question"], date_fields=["created_at"])

        index.append({"question": "How do I use Python?", "created_at": date(2024, 1, 15)})
        index.append({"question": "What is ML with Python?", "created_at": date(2024, 2, 20)})
        index.append({"question": "How to write tests?", "created_at": date(2024, 3, 10)})

        results = index.search("python", filter_dict={"created_at": [('>=', date(2024, 2, 1))]})
        assert all(doc["created_at"] >= date(2024, 2, 1) for doc in results)

    def test_fit_then_append_with_numeric_field(self):
        """Test fit followed by append with numeric field."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "price": 100},
            {"question": "What is machine learning?", "text": "ML is AI.", "price": 200},
        ]

        index = AppendableIndex(text_fields=["question", "text"], numeric_fields=["price"])
        index.fit(docs)

        # Append more docs
        index.append({"question": "How to write tests in Python?", "text": "Tests.", "price": 150})
        index.append({"question": "What is data science?", "text": "Data.", "price": 50})

        # All docs should be searchable with numeric filter
        results = index.search("python", filter_dict={"price": [('>=', 100)]})
        assert all(doc["price"] >= 100 for doc in results)

    def test_all_numeric_operators(self):
        """Test all supported numeric operators."""
        docs = [
            {"question": "Doc 1", "text": "Test", "value": 10},
            {"question": "Doc 2", "text": "Test", "value": 20},
            {"question": "Doc 3", "text": "Test", "value": 30},
        ]

        index = AppendableIndex(text_fields=["question"], numeric_fields=["value"])
        index.fit(docs)

        # Test >=
        results = index.search("test", filter_dict={"value": [('>=', 20)]})
        assert all(doc["value"] >= 20 for doc in results)

        # Test >
        results = index.search("test", filter_dict={"value": [('>', 20)]})
        assert all(doc["value"] > 20 for doc in results)

        # Test <=
        results = index.search("test", filter_dict={"value": [('<=', 20)]})
        assert all(doc["value"] <= 20 for doc in results)

        # Test <
        results = index.search("test", filter_dict={"value": [('<', 20)]})
        assert all(doc["value"] < 20 for doc in results)

        # Test ==
        results = index.search("test", filter_dict={"value": [('==', 20)]})
        assert all(doc["value"] == 20 for doc in results)

        # Test !=
        results = index.search("test", filter_dict={"value": [('!=', 20)]})
        assert all(doc["value"] != 20 for doc in results)

    def test_numeric_exact_match_equality(self):
        """Test numeric field exact match with simple value (not list)."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "price": 100},
            {"question": "What is machine learning?", "text": "ML is AI.", "price": 200},
            {"question": "How to write tests?", "text": "Tests with Python.", "price": 150},
            {"question": "What is data science?", "text": "Data science.", "price": 50},
        ]

        index = AppendableIndex(text_fields=["question", "text"], numeric_fields=["price"])
        index.fit(docs)

        # Exact match: price == 150
        results = index.search("python", filter_dict={"price": 150})
        assert len(results) == 1
        assert results[0]["price"] == 150

    def test_date_exact_match_equality(self):
        """Test date field exact match with simple value (not list)."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "created_at": date(2024, 1, 15)},
            {"question": "What is ML with Python?", "text": "ML is AI.", "created_at": date(2024, 2, 20)},
            {"question": "How to write tests?", "text": "Tests with Python.", "created_at": date(2024, 3, 10)},
            {"question": "What is data science?", "text": "Data science.", "created_at": date(2024, 1, 5)},
        ]

        index = AppendableIndex(text_fields=["question", "text"], date_fields=["created_at"])
        index.fit(docs)

        # Exact match: created_at == date(2024, 2, 20)
        results = index.search("python", filter_dict={"created_at": date(2024, 2, 20)})
        assert len(results) == 1
        assert results[0]["created_at"] == date(2024, 2, 20)

    def test_numeric_none_values(self):
        """Test numeric field with None values."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "price": 100},
            {"question": "What is ML with Python?", "text": "ML is AI.", "price": None},
            {"question": "How to write tests?", "text": "Tests with Python.", "price": 200},
        ]

        index = AppendableIndex(text_fields=["question", "text"], numeric_fields=["price"])
        index.fit(docs)

        # price == None
        results = index.search("python", filter_dict={"price": None})
        assert len(results) == 1
        assert results[0]["price"] is None

    def test_date_none_values(self):
        """Test date field with None values."""
        docs = [
            {"question": "How do I use Python?", "text": "Python is great.", "created_at": date(2024, 1, 15)},
            {"question": "What is ML with Python?", "text": "ML is AI.", "created_at": None},
            {"question": "How to write tests?", "text": "Tests with Python.", "created_at": date(2024, 2, 20)},
        ]

        index = AppendableIndex(text_fields=["question", "text"], date_fields=["created_at"])
        index.fit(docs)

        # created_at == None
        results = index.search("python", filter_dict={"created_at": None})
        assert len(results) == 1
        assert results[0]["created_at"] is None

    def test_multiple_numeric_fields(self):
        """Test filtering on multiple numeric fields simultaneously."""
        docs = [
            {"question": "How do I use Python?", "text": "Python.", "price": 100, "rating": 4.5},
            {"question": "What is machine learning?", "text": "ML.", "price": 200, "rating": 3.8},
            {"question": "How to write tests in Python?", "text": "Tests.", "price": 150, "rating": 4.2},
            {"question": "What is data science?", "text": "Data.", "price": 50, "rating": 4.8},
        ]

        index = AppendableIndex(text_fields=["question", "text"], numeric_fields=["price", "rating"])
        index.fit(docs)

        # price >= 100 AND rating >= 4.0
        results = index.search("python", filter_dict={
            "price": [('>=', 100)],
            "rating": [('>=', 4.0)]
        })
        assert all(doc["price"] >= 100 and doc["rating"] >= 4.0 for doc in results)

    def test_zero_values_in_numeric_fields(self):
        """Test that zero values are handled correctly in numeric fields."""
        docs = [
            {"question": "Test doc one", "text": "Test", "value": 0},
            {"question": "Test doc two", "text": "Test", "value": 10},
            {"question": "Test doc three", "text": "Test", "value": -5},
        ]

        index = AppendableIndex(text_fields=["question", "text"], numeric_fields=["value"])
        index.fit(docs)

        # value == 0
        results = index.search("test", filter_dict={"value": 0})
        assert len(results) == 1
        assert results[0]["value"] == 0

        # value >= 0
        results = index.search("test", filter_dict={"value": [('>=', 0)]})
        assert all(doc["value"] >= 0 for doc in results)

    def test_negative_numeric_values(self):
        """Test that negative values work correctly in numeric fields."""
        docs = [
            {"question": "Doc 1", "text": "Test", "value": -10},
            {"question": "Doc 2", "text": "Test", "value": -5},
            {"question": "Doc 3", "text": "Test", "value": 0},
        ]

        index = AppendableIndex(text_fields=["question"], numeric_fields=["value"])
        index.fit(docs)

        # value >= -7
        results = index.search("test", filter_dict={"value": [('>=', -7)]})
        assert all(doc["value"] >= -7 for doc in results)

        # -10 < value < 0
        results = index.search("test", filter_dict={"value": [('>', -10), ('<', 0)]})
        assert all(doc["value"] > -10 and doc["value"] < 0 for doc in results)

    def test_combined_equality_and_range_filters(self):
        """Test mixing equality (simple value) and range (list) filters."""
        docs = [
            {"question": "How do I use Python?", "text": "Python.", "price": 100, "category": "A"},
            {"question": "What is machine learning?", "text": "ML.", "price": 200, "category": "B"},
            {"question": "How to write tests in Python?", "text": "Tests.", "price": 150, "category": "A"},
            {"question": "What is data science?", "text": "Data.", "price": 50, "category": "A"},
        ]

        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["category"],
            numeric_fields=["price"]
        )
        index.fit(docs)

        # category == "A" (equality) AND price >= 100 (range)
        results = index.search("python", filter_dict={
            "category": "A",
            "price": [('>=', 100)]
        })
        assert all(doc["category"] == "A" and doc["price"] >= 100 for doc in results)

    def test_append_with_numeric_exact_match(self):
        """Test append with numeric exact match filter."""
        index = AppendableIndex(text_fields=["question"], numeric_fields=["price"])

        index.append({"question": "How do I use Python?", "price": 100})
        index.append({"question": "What is ML with Python?", "price": 200})
        index.append({"question": "How to write tests in Python?", "price": 200})

        # Exact match: price == 200
        results = index.search("python", filter_dict={"price": 200})
        assert len(results) == 2
        assert all(doc["price"] == 200 for doc in results)

    def test_append_with_date_exact_match(self):
        """Test append with date exact match filter."""
        index = AppendableIndex(text_fields=["question"], date_fields=["created_at"])

        target_date = date(2024, 2, 1)
        index.append({"question": "How do I use Python?", "created_at": date(2024, 1, 15)})
        index.append({"question": "What is ML with Python?", "created_at": target_date})
        index.append({"question": "How to write Python tests?", "created_at": target_date})

        # Exact match: created_at == target_date
        results = index.search("python", filter_dict={"created_at": target_date})
        assert len(results) == 2
        assert all(doc["created_at"] == target_date for doc in results)
