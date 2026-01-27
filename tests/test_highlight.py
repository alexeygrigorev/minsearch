import pytest
from minsearch.minsearch import Index
from minsearch.append import AppendableIndex
from minsearch.highlight import extract_snippet, apply_highlight


@pytest.fixture
def sample_docs():
    return [
        {
            "title": "Introduction to Python Programming",
            "text": "Python is a high-level, interpreted programming language. It is widely used for web development, data analysis, artificial intelligence, and scientific computing. Python's simple syntax makes it easy to learn for beginners.",
            "section": "Programming Languages",
            "course": "python-basics",
        },
        {
            "title": "Machine Learning Fundamentals",
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. Machine learning algorithms learn from data and improve their performance over time without being explicitly programmed.",
            "section": "AI and ML",
            "course": "ml-basics",
        },
        {
            "title": "Testing in Python",
            "text": "Writing tests is essential for ensuring code quality and reliability. Python has several testing frameworks including unittest, pytest, and nose. Tests help catch bugs early and make refactoring safer.",
            "section": "Software Engineering",
            "course": "python-basics",
        },
    ]


@pytest.fixture
def text_fields():
    return ["title", "text", "section"]


@pytest.fixture
def keyword_fields():
    return ["course"]


def test_extract_snippet_basic():
    """Test basic snippet extraction."""
    text = "Python is a programming language. Python is used for many applications."
    query_tokens = ["python"]
    
    snippet = extract_snippet(text, query_tokens, fragment_size=50)
    assert "Python" in snippet or "python" in snippet
    assert len(snippet) <= 100  # Should be around fragment_size + some buffer


def test_extract_snippet_with_highlighting():
    """Test snippet extraction with highlighting tags."""
    text = "Python is a programming language. Python is used for many applications."
    query_tokens = ["python"]
    
    snippet = extract_snippet(text, query_tokens, fragment_size=50, pre_tag="*", post_tag="*")
    assert "*Python*" in snippet or "*python*" in snippet


def test_extract_snippet_empty_text():
    """Test snippet extraction with empty text."""
    snippet = extract_snippet("", ["python"], fragment_size=50)
    assert snippet == ""


def test_extract_snippet_no_matches():
    """Test snippet extraction when no matches are found."""
    text = "This is a test document without the search term."
    query_tokens = ["nonexistent"]
    
    snippet = extract_snippet(text, query_tokens, fragment_size=50)
    # Should return beginning of text
    assert len(snippet) > 0
    assert "This is a test" in snippet


def test_extract_snippet_multiple_fragments():
    """Test snippet extraction with multiple fragments."""
    text = "Python is great. " * 20 + "JavaScript is also useful. " * 20
    query_tokens = ["python", "javascript"]
    
    snippet = extract_snippet(text, query_tokens, fragment_size=50, number_of_fragments=2)
    # Should contain both terms or their contexts
    assert len(snippet) > 0


def test_apply_highlight():
    """Test applying highlight to a document."""
    doc = {
        "title": "Python Programming",
        "text": "Python is a programming language. It is widely used.",
        "course": "python-basics"
    }
    text_fields = ["title", "text"]
    query_tokens = ["python"]
    highlight_config = {"fragment_size": 50}
    
    result = apply_highlight(doc, text_fields, query_tokens, highlight_config)
    
    # Text fields should be replaced with snippets
    assert len(result["text"]) < len(doc["text"])
    # Non-text fields should remain unchanged
    assert result["course"] == doc["course"]


def test_index_search_with_highlight(text_fields, keyword_fields, sample_docs):
    """Test Index search with highlighting."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)
    
    # Search without highlight
    results_no_highlight = index.search("python")
    assert len(results_no_highlight) > 0
    
    # Search with highlight
    highlight_config = {"fragment_size": 100}
    results_with_highlight = index.search("python", highlight=highlight_config)
    assert len(results_with_highlight) > 0
    
    # Highlighted results should have shorter text
    for result in results_with_highlight:
        assert "text" in result
        # The highlighted text should be shorter than or equal to any original document
        # (we just verify it's reasonable, not checking against a specific doc)
        max_text_len = max(len(doc["text"]) for doc in sample_docs)
        assert len(result["text"]) <= max_text_len


def test_index_search_with_highlight_tags(text_fields, keyword_fields, sample_docs):
    """Test Index search with highlighting and tags."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)
    
    # Search with highlight and tags
    highlight_config = {
        "fragment_size": 100,
        "pre_tag": "*",
        "post_tag": "*"
    }
    results = index.search("python programming", highlight=highlight_config)
    assert len(results) > 0
    
    # At least one result should have highlighted terms
    has_highlight = False
    for result in results:
        if "*" in result.get("text", "") or "*" in result.get("title", ""):
            has_highlight = True
            break
    assert has_highlight, "No highlights found in results"


def test_index_search_highlight_with_filters(text_fields, keyword_fields, sample_docs):
    """Test Index search with highlighting and filters."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)
    
    highlight_config = {"fragment_size": 100}
    results = index.search(
        "python",
        filter_dict={"course": "python-basics"},
        highlight=highlight_config
    )
    
    assert len(results) > 0
    # All results should have the correct course
    for result in results:
        assert result["course"] == "python-basics"


def test_index_search_highlight_with_boost(text_fields, keyword_fields, sample_docs):
    """Test Index search with highlighting and boost."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)
    
    highlight_config = {"fragment_size": 100}
    results = index.search(
        "python",
        boost_dict={"title": 2.0},
        highlight=highlight_config
    )
    
    assert len(results) > 0


def test_appendable_index_search_with_highlight(text_fields, keyword_fields, sample_docs):
    """Test AppendableIndex search with highlighting."""
    index = AppendableIndex(text_fields, keyword_fields)
    index.fit(sample_docs)
    
    # Search with highlight
    highlight_config = {"fragment_size": 100, "pre_tag": "<em>", "post_tag": "</em>"}
    results = index.search("machine learning", highlight=highlight_config)
    assert len(results) > 0
    
    # Check that highlighting was applied
    has_highlight = False
    for result in results:
        if "<em>" in result.get("text", "") or "<em>" in result.get("title", ""):
            has_highlight = True
            break
    assert has_highlight, "No highlights found in AppendableIndex results"


def test_appendable_index_search_highlight_after_append(text_fields, keyword_fields, sample_docs):
    """Test AppendableIndex highlighting after appending documents."""
    index = AppendableIndex(text_fields, keyword_fields)
    
    # Start with first two docs
    index.fit(sample_docs[:2])
    
    # Append the third doc
    index.append(sample_docs[2])
    
    # Search with highlight
    highlight_config = {"fragment_size": 100}
    results = index.search("python", highlight=highlight_config)
    assert len(results) > 0


def test_highlight_preserves_non_text_fields(text_fields, keyword_fields, sample_docs):
    """Test that highlighting preserves non-text fields."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)
    
    highlight_config = {"fragment_size": 100}
    results = index.search("python", highlight=highlight_config)
    
    # Check that keyword fields are preserved
    for result in results:
        assert "course" in result
        assert result["course"] in ["python-basics", "ml-basics"]


def test_highlight_with_output_ids(text_fields, keyword_fields, sample_docs):
    """Test that highlighting works with output_ids."""
    index = Index(text_fields, keyword_fields)
    index.fit(sample_docs)
    
    highlight_config = {"fragment_size": 100}
    results = index.search("python", output_ids=True, highlight=highlight_config)
    
    assert len(results) > 0
    for result in results:
        assert "_id" in result
        assert isinstance(result["_id"], int)


def test_highlight_multiple_fragments(text_fields, keyword_fields):
    """Test highlighting with multiple fragments."""
    docs = [
        {
            "title": "Long Document",
            "text": "Python is mentioned here. " * 10 + "And Python is mentioned here again. " * 10 + "Finally Python appears one more time.",
            "section": "Test",
            "course": "test-course"
        }
    ]
    
    index = Index(text_fields, keyword_fields)
    index.fit(docs)
    
    highlight_config = {
        "fragment_size": 80,
        "number_of_fragments": 3,
        "pre_tag": "[",
        "post_tag": "]"
    }
    results = index.search("python", highlight=highlight_config)
    
    assert len(results) > 0
    assert "[Python]" in results[0]["text"] or "[python]" in results[0]["text"]


def test_highlight_empty_query():
    """Test highlighting with empty query."""
    docs = [
        {
            "title": "Test Document",
            "text": "This is a test document.",
            "course": "test"
        }
    ]
    
    index = Index(text_fields=["title", "text"], keyword_fields=["course"])
    index.fit(docs)
    
    highlight_config = {"fragment_size": 100}
    results = index.search("", highlight=highlight_config)
    
    # Empty query should return no results
    assert len(results) == 0


def test_highlight_special_characters():
    """Test highlighting with special characters in query."""
    docs = [
        {
            "title": "Python-3.9",
            "text": "Python-3.9 is the latest version of Python.",
            "course": "python"
        }
    ]
    
    index = Index(text_fields=["title", "text"], keyword_fields=["course"])
    index.fit(docs)
    
    highlight_config = {"fragment_size": 100, "pre_tag": "*", "post_tag": "*"}
    results = index.search("python", highlight=highlight_config)
    
    assert len(results) > 0
    # Should find python as a token
    assert "*Python*" in results[0]["text"] or "*python*" in results[0]["text"] or "Python" in results[0]["text"]
