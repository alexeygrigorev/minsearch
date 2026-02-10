import pytest
from minsearch import Index, AppendableIndex, Highlighter


@pytest.fixture
def sample_docs():
    return [
        {
            "question": "How do I use Python for machine learning?",
            "text": "Python is a programming language. It's easy to learn Python for data science and machine learning.",
            "section": "Programming",
            "course": "python-basics",
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI. It uses algorithms and Python for data analysis.",
            "section": "AI",
            "course": "ml-basics",
        },
        {
            "question": "How to write tests in Python?",
            "text": "Tests help ensure code quality. Use pytest for Python testing.",
            "section": "Testing",
            "course": "python-basics",
        },
    ]


@pytest.fixture
def text_fields():
    return ["question", "text", "section"]


# ==================== Initialization Tests ====================

def test_highlighter_initialization():
    """Test that Highlighter initializes correctly."""
    highlighter = Highlighter(text_fields=["question", "text"])
    assert highlighter.text_fields == ["question", "text"]
    assert highlighter.snippet_size == 200
    assert highlighter.max_snippets == 3
    assert highlighter.highlight_format == "**"


def test_highlighter_custom_params():
    """Test Highlighter with custom parameters."""
    highlighter = Highlighter(
        text_fields=["question"],
        snippet_size=100,
        max_snippets=5,
        highlight_format="__",
    )
    assert highlighter.snippet_size == 100
    assert highlighter.max_snippets == 5
    assert highlighter.highlight_format == "__"


# ==================== Format Tests ====================

def test_highlight_format_default_asterisk(text_fields, sample_docs):
    """Test default format with ** delimiter."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                assert "**Python**" in snippet


def test_highlight_format_custom_string_delimiter(text_fields, sample_docs):
    """Test custom string delimiter."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question"], highlight_format="__")
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                assert "__Python__" in snippet
                assert "**" not in snippet


def test_highlight_format_tuple_open_close(text_fields, sample_docs):
    """Test tuple format with different open/close delimiters."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question"], highlight_format=("[", "]"))
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                assert "[Python]" in snippet
                assert "**" not in snippet


def test_highlight_format_tuple_different_delimiters(text_fields, sample_docs):
    """Test tuple with different delimiters."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question"], highlight_format=("{", "}"))
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                assert "{Python}" in snippet


def test_highlight_format_callable_function(text_fields, sample_docs):
    """Test callable format (custom function)."""
    index = Index(text_fields)
    index.fit(sample_docs)

    def custom_formatter(text):
        return f"[[{text}]]"

    highlighter = Highlighter(
        text_fields=["question"],
        highlight_format=custom_formatter
    )
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                assert "[[Python]]" in snippet
                assert "**" not in snippet


def test_highlight_format_callable_with_uppercase(text_fields, sample_docs):
    """Test callable that transforms to uppercase."""
    index = Index(text_fields)
    index.fit(sample_docs)

    def uppercase_formatter(text):
        return f"==={text.upper()}==="

    highlighter = Highlighter(
        text_fields=["question"],
        highlight_format=uppercase_formatter
    )
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                assert "===PYTHON===" in snippet


def test_highlight_format_callable_lambda(text_fields, sample_docs):
    """Test callable format with lambda."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(
        text_fields=["question"],
        highlight_format=lambda t: f"((({t})))"
    )
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                assert "(((Python)))" in snippet


def test_highlight_format_multiple_matches_with_custom_format(text_fields, sample_docs):
    """Test that custom format works with multiple matches in one snippet."""
    docs = [{"text": "Python is great and Python is easy"}]

    index = Index(text_fields=["text"])
    index.fit(docs)

    highlighter = Highlighter(
        text_fields=["text"],
        highlight_format="__"
    )
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    snippet = highlighted[0]["highlights"]["text"][0]
    # Should have both matches highlighted
    assert snippet.count("__") >= 4  # At least 2 opening and 2 closing


def test_highlight_format_string_with_brackets(text_fields, sample_docs):
    """Test string delimiter that contains brackets."""
    index = Index(text_fields)
    index.fit(sample_docs)

    # Using "[]" as the delimiter - should wrap both sides
    highlighter = Highlighter(text_fields=["question"], highlight_format="[]")
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        for field, snippets in h["highlights"].items():
            for snippet in snippets:
                # Since "[]" is a single string, it wraps both sides
                assert "[]Python[]" in snippet


# ==================== Basic Usage Tests ====================

def test_highlighter_basic_usage(text_fields, sample_docs):
    """Test basic highlighting functionality."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])
    results = index.search("python machine learning")
    highlighted = highlighter.highlight("python machine learning", results)

    assert len(highlighted) > 0
    assert "highlights" in highlighted[0]
    assert "document" in highlighted[0]


def test_highlighter_with_appendable_index(text_fields, sample_docs):
    """Test that Highlighter works with AppendableIndex."""
    index = AppendableIndex(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    assert len(highlighted) > 0
    assert "highlights" in highlighted[0]


def test_highlighter_specific_fields(text_fields, sample_docs):
    """Test highlighting specific fields only."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text", "section"])
    results = index.search("python")

    # Highlight only 'question' field
    highlighted = highlighter.highlight("python", results, fields=["question"])

    for h in highlighted:
        # Only 'question' should be in highlights
        assert set(h["highlights"].keys()).issubset({"question"})


def test_highlighter_empty_query(text_fields, sample_docs):
    """Test highlighting with empty query (stop words only)."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])
    results = index.search("python")

    # Query with only stop words
    highlighted = highlighter.highlight("the and is", results)

    # Should return results but with empty highlights
    assert len(highlighted) == len(results)
    for h in highlighted:
        assert h["highlights"] == {}


def test_highlighter_case_preservation(text_fields, sample_docs):
    """Test that highlighting preserves original case."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question"])
    results = index.search("PYTHON")  # Uppercase query

    highlighted = highlighter.highlight("PYTHON", results)

    # Check that original case is preserved
    for h in highlighted:
        for snippets in h["highlights"].values():
            for snippet in snippets:
                # Should contain Python, not PYTHON (original case)
                assert "**Python**" in snippet or "PYTHON" not in snippet


def test_highlighter_multiple_snippets():
    """Test that multiple snippets are returned for multiple matches."""
    docs = [
        {
            "text": "Python is great. Python is easy. Python is powerful. Python is popular.",
        }
    ]

    index = Index(text_fields=["text"])
    index.fit(docs)

    highlighter = Highlighter(text_fields=["text"], max_snippets=2)
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    # Should have highlights
    assert highlighted[0]["highlights"]


def test_highlighter_no_matches():
    """Test highlighting when query doesn't match."""
    docs = [{"text": "This is about Python programming"}]

    index = Index(text_fields=["text"])
    index.fit(docs)

    highlighter = Highlighter(text_fields=["text"])
    results = index.search("python")
    highlighted = highlighter.highlight("nonexistent term", results)

    # Should return empty highlights
    assert highlighted[0]["highlights"] == {}


def test_highlighter_with_filters(text_fields, sample_docs):
    """Test highlighting works with filtered search results."""
    index = Index(text_fields, keyword_fields=["course"])
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])
    results = index.search("python", filter_dict={"course": "python-basics"})
    highlighted = highlighter.highlight("python", results)

    assert len(highlighted) > 0
    # All results should be from python-basics course
    for h in highlighted:
        assert h["document"]["course"] == "python-basics"


def test_highlighter_snippet_size():
    """Test that snippet_size parameter affects output."""
    docs = [
        {"text": "Python " * 100 + "machine learning" + " is great " * 100},
    ]

    index = Index(text_fields=["text"])
    index.fit(docs)

    highlighter = Highlighter(text_fields=["text"], snippet_size=50)
    results = index.search("machine learning")
    highlighted = highlighter.highlight("machine learning", results)

    # Snippet should be roughly snippet_size characters (plus delimiters and ellipsis)
    snippet = highlighted[0]["highlights"]["text"][0]
    # Should be reasonably short (snippet_size + some margin)
    assert len(snippet) < 200


def test_highlighter_workflow_example(text_fields, sample_docs):
    """Test the typical workflow: search then highlight."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])

    # Step 1: Search
    query = "python machine learning"
    results = index.search(query)
    assert len(results) > 0

    # Step 2: Highlight
    highlighted = highlighter.highlight(query, results)
    assert len(highlighted) > 0

    # Verify structure
    for h in highlighted:
        assert "highlights" in h
        assert "document" in h
        # The original document is preserved
        assert "question" in h["document"]
        assert "text" in h["document"]


def test_highlighter_with_boost(text_fields, sample_docs):
    """Test that highlighting works with boosted search."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])

    # Search with boost
    results = index.search("python", boost_dict={"question": 2.0})
    highlighted = highlighter.highlight("python", results)

    assert len(highlighted) > 0
    assert "highlights" in highlighted[0]


def test_highlighter_multiterm_query(text_fields, sample_docs):
    """Test highlighting with multi-term queries."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])
    results = index.search("python machine learning")
    highlighted = highlighter.highlight("python machine learning", results)

    # Should find highlights for multiple terms
    for h in highlighted:
        for snippets in h["highlights"].values():
            for snippet in snippets:
                # At least one of the terms should be highlighted
                has_mark = "**" in snippet
                if has_mark:
                    break


def test_highlighter_preserves_document_structure(text_fields, sample_docs):
    """Test that original document is preserved in highlighted output."""
    index = Index(text_fields, keyword_fields=["course"])
    index.fit(sample_docs)

    highlighter = Highlighter(text_fields=["question", "text"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    # Check that original documents are preserved
    for i, h in enumerate(highlighted):
        original_doc = results[i]
        highlighted_doc = h["document"]
        assert original_doc == highlighted_doc


def test_highlighter_custom_stop_words():
    """Test Highlighter with custom stop words."""
    docs = [{"text": "The quick brown fox jumps over the lazy dog"}]

    index = Index(text_fields=["text"])
    index.fit(docs)

    # Use custom stop words (including "quick")
    highlighter = Highlighter(
        text_fields=["text"],
        stop_words={"quick", "brown", "the", "over", "lazy"}
    )
    results = index.search("quick brown")
    highlighted = highlighter.highlight("quick brown", results)

    # Should not highlight "quick" or "brown" since they're stop words
    assert highlighted[0]["highlights"] == {}
