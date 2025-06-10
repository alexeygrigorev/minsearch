import pytest
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

    # Test non-existent filter
    results = index.search("programming", filter_dict={"course": "non_existent"})
    assert len(results) == 0

    # Test multiple filters
    results = index.search(
        "programming", filter_dict={"course": "CS101", "non_existent": "value"}
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

    # Test with multiple filters
    results = index.search(
        "python", filter_dict={"course": "CS101", "non_existent": "value"}
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
