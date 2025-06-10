import pytest
from minsearch.minsearch import Index


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
    """Test that index initializes correctly with different parameters."""
    # Test with default parameters
    index = Index(text_fields, keyword_fields)
    assert index.text_fields == text_fields
    assert index.keyword_fields == keyword_fields

    # Test with custom vectorizer parameters
    vectorizer_params = {"max_features": 1000, "min_df": 2, "max_df": 0.95}
    index = Index(text_fields, keyword_fields, vectorizer_params)
    assert index.text_fields == text_fields
    assert index.keyword_fields == keyword_fields


def test_fit_and_search(text_fields, keyword_fields, sample_docs):
    """Test that fit and search work correctly together."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)

    # Test basic search
    results = index.search("python")
    assert len(results) > 0
    assert any("python" in doc["text"].lower() for doc in results)

    # Test search with filters
    results = index.search("python", filter_dict={"course": "python-basics"})
    assert len(results) > 0
    assert all(doc["course"] == "python-basics" for doc in results)

    # Test search with boosts
    results = index.search("python", boost_dict={"question": 2, "text": 1})
    assert len(results) > 0


def test_search_ranking(text_fields, keyword_fields, sample_docs):
    """Test that search results are properly ranked."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)

    # Search for a term that appears in both question and text
    results = index.search("python")

    # Results should be ordered by relevance
    if len(results) > 1:
        # Check that documents with the term in the question appear first
        # when question field is boosted
        boosted_results = index.search("python", boost_dict={"question": 2, "text": 1})
        assert len(boosted_results) > 0


def test_empty_docs(text_fields, keyword_fields):
    """Test behavior with empty document list."""
    index = Index(text_fields, keyword_fields)
    index.fit([])

    # Search should return empty list
    assert len(index.search("python")) == 0


def test_empty_fields(text_fields, keyword_fields, sample_docs):
    """Test behavior with empty field values."""
    # Add a doc with empty fields
    docs_with_empty = sample_docs + [
        {"question": "", "text": "", "section": "", "course": ""}
    ]

    index = Index(text_fields, keyword_fields)
    index.fit(docs_with_empty)

    # Search should still work
    results = index.search("python")
    assert len(results) > 0


def test_boost_weights(text_fields, keyword_fields, sample_docs):
    """Test that boost weights affect search results."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)

    # Search with different boost weights
    boost_configs = [
        {"question": 1, "text": 1, "section": 1},
        {"question": 2, "text": 1, "section": 1},
        {"question": 1, "text": 2, "section": 1},
        {"question": 1, "text": 1, "section": 2},
    ]

    for boost_dict in boost_configs:
        results = index.search("python", boost_dict=boost_dict)
        assert len(results) > 0


def test_filter_combinations(text_fields, keyword_fields, sample_docs):
    """Test different combinations of filters."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)

    results = index.search("python", filter_dict={"course": "python-basics"})
    assert len(results) > 0, "No results found for python-basics"
    assert all(doc["course"] == "python-basics" for doc in results)

    results = index.search("python", filter_dict={"course": "nonexistent"})
    assert len(results) == 0

    results = index.search("machine learning", filter_dict={"course": "ml-basics"})
    assert len(results) > 0, "No results found for ml-basics"
    assert all(doc["course"] == "ml-basics" for doc in results)


def test_num_results(text_fields, keyword_fields, sample_docs):
    """Test that num_results parameter works correctly."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)

    # Test with different num_results values
    for n in [1, 2, 5, 10]:
        results = index.search("python", num_results=n)
        assert len(results) <= n


def test_basic_search():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = Index(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    results = index.search("python")
    assert len(results) > 0
    assert results[0]["title"] == "Python Programming"


def test_filter_combinations():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = Index(text_fields=["title"], keyword_fields=["course"])
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
    index = Index(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    # Test without boost
    results = index.search("python")
    assert len(results) > 0

    # Test with boost
    results = index.search("python", boost_dict={"title": 2.0})
    assert len(results) > 0


def test_empty_docs():
    index = Index(text_fields=["title"], keyword_fields=["course"])
    index.fit([])

    results = index.search("python")
    assert len(results) == 0


def test_vectorizer_parameters():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]

    # Test with custom parameters
    index = Index(
        text_fields=["title"],
        keyword_fields=["course"],
        vectorizer_params={"min_df": 1, "max_df": 1.0},
    )
    index.fit(docs)

    results = index.search("python")
    assert len(results) > 0


def test_multiple_text_fields():
    docs = [
        {"title": "Python Programming", "description": "Learn Python programming"},
        {"title": "Data Science", "description": "Python for data science"},
    ]
    index = Index(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    results = index.search("python")
    assert len(results) > 0


def test_keyword_fields():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = Index(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    results = index.search("programming", filter_dict={"course": "CS101"})
    assert len(results) > 0
    assert results[0]["course"] == "CS101"


def test_num_results():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = Index(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    results = index.search("python", num_results=1)
    assert len(results) == 1


def test_readme_example():
    # Test the exact example from README
    docs = [
        {
            "title": "Python Programming",
            "description": "Learn Python programming",
            "course": "CS101",
        },
        {
            "title": "Data Science",
            "description": "Python for data science",
            "course": "CS102",
        },
        {
            "title": "Machine Learning",
            "description": "Introduction to ML",
            "course": "CS103",
        },
    ]

    index = Index(text_fields=["title", "description"], keyword_fields=["course"])
    index.fit(docs)

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


def test_stop_words():
    docs = [
        {
            "title": "The Python Programming",
            "description": "Learn the Python programming",
        },
        {"title": "Data Science", "description": "Python for data science"},
    ]

    # Test with default stop words
    index = Index(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    results = index.search("the python")
    assert len(results) > 0

    # Test with custom stop words
    index = Index(
        text_fields=["title", "description"],
        keyword_fields=[],
        vectorizer_params={"stop_words": ["the", "a", "an"]},
    )
    index.fit(docs)

    results = index.search("the python")
    assert len(results) > 0


def test_empty_query():
    docs = [
        {"title": "Python Programming", "course": "CS101"},
        {"title": "Data Science", "course": "CS102"},
    ]
    index = Index(text_fields=["title"], keyword_fields=["course"])
    index.fit(docs)

    results = index.search("")
    assert len(results) == 0


def test_special_characters():
    docs = [
        {"title": "Python-Programming", "description": "Learn Python (programming)"},
        {"title": "Data-Science", "description": "Python for data-science"},
    ]
    index = Index(text_fields=["title", "description"], keyword_fields=[])
    index.fit(docs)

    results = index.search("python-programming")
    assert len(results) > 0
    assert results[0]["title"] == "Python-Programming"


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

    index = Index(text_fields=["title", "description"], keyword_fields=["course"])
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
    index = Index(text_fields, keyword_fields)
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
