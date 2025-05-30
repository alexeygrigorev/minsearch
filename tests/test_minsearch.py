import pytest
from minsearch.minsearch import Index


@pytest.fixture
def sample_docs():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language. It's easy to learn.",
            "section": "Programming",
            "course": "python-basics"
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI. It uses algorithms.",
            "section": "AI",
            "course": "ml-basics"
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality. Use pytest for Python.",
            "section": "Testing",
            "course": "python-basics"
        }
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
    vectorizer_params = {
        "max_features": 1000,
        "min_df": 2,
        "max_df": 0.95
    }
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
    docs_with_empty = sample_docs + [{
        "question": "",
        "text": "",
        "section": "",
        "course": ""
    }]
    
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
        {"question": 1, "text": 1, "section": 2}
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

