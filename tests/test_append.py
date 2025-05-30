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
    queries = [
        "python",
        "machine learning",
        "testing",
        "nonexistent"
    ]
    
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
    queries = [
        "python",
        "machine learning",
        "testing",
        "nonexistent"
    ]
    
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
    docs_with_empty = sample_docs + [{
        "question": "",
        "text": "",
        "section": "",
        "course": ""
    }]
    
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
        {"question": 1, "text": 1, "section": 2}
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
        {"course": "nonexistent"}
    ]
    
    for filter_dict in filter_configs:
        index_results = index.search("python", filter_dict=filter_dict)
        appendable_results = appendable_index.search("python", filter_dict=filter_dict)
        assert len(index_results) == len(appendable_results) 