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
        {
            "question": "I just discovered the course, can I still join?",
            "text": "Yes, you can still join the course even after discovering it late. You'll have access to all course materials.",
            "section": "Enrollment",
            "course": "general",
        },
    ]


@pytest.fixture
def text_fields():
    return ["question", "text", "section"]


# ==================== Initialization Tests ====================

def test_highlighter_initialization():
    """Test that Highlighter initializes correctly."""
    highlighter = Highlighter(highlight_fields=["question", "text"])
    assert highlighter.highlight_fields == ["question", "text"]
    assert highlighter.skip_fields == set()
    assert highlighter.max_matches == 5
    assert highlighter.highlight_format == "**"


def test_highlighter_with_skip_fields():
    """Test Highlighter with skip fields."""
    highlighter = Highlighter(
        highlight_fields=["question", "text"],
        skip_fields=["course"]
    )
    assert highlighter.highlight_fields == ["question", "text"]
    assert highlighter.skip_fields == {"course"}


def test_highlighter_custom_params():
    """Test Highlighter with custom parameters."""
    highlighter = Highlighter(
        highlight_fields=["question"],
        skip_fields=["course", "section"],
        max_matches=3,
        snippet_size=100,
        highlight_format="__",
    )
    assert highlighter.max_matches == 3
    assert highlighter.snippet_size == 100
    assert highlighter.highlight_format == "__"
    assert highlighter.skip_fields == {"course", "section"}


# ==================== Format Tests ====================

def test_highlight_format_default_asterisk(text_fields, sample_docs):
    """Test default format with ** delimiter."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        assert "question" in h
        assert "matches" in h["question"]
        assert "total_matches" in h["question"]
        if h["question"]["matches"]:
            assert "**Python**" in h["question"]["matches"][0]


def test_highlight_format_custom_string_delimiter(text_fields, sample_docs):
    """Test custom string delimiter."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question"], highlight_format="__")
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        if h["question"]["matches"]:
            assert "__Python__" in h["question"]["matches"][0]
            assert "**" not in h["question"]["matches"][0]


def test_highlight_format_tuple_open_close(text_fields, sample_docs):
    """Test tuple format with different open/close delimiters."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question"], highlight_format=("[", "]"))
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        if h["question"]["matches"]:
            assert "[Python]" in h["question"]["matches"][0]


def test_highlight_format_callable(text_fields, sample_docs):
    """Test callable format."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(
        highlight_fields=["question"],
        highlight_format=lambda t: f"==={t.upper()}==="
    )
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        if h["question"]["matches"]:
            assert "===PYTHON===" in h["question"]["matches"][0]


# ==================== Basic Usage Tests ====================

def test_highlighter_basic_usage(text_fields, sample_docs):
    """Test basic highlighting functionality."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])
    results = index.search("python machine learning")
    highlighted = highlighter.highlight("python machine learning", results)

    assert len(highlighted) > 0
    h = highlighted[0]
    assert "question" in h
    assert "text" in h
    assert "matches" in h["question"]
    assert "total_matches" in h["question"]


def test_highlighter_with_skip_fields_excludes_fields(text_fields, sample_docs):
    """Test that skip_fields are excluded from output."""
    index = Index(text_fields, keyword_fields=["course"])
    index.fit(sample_docs)

    highlighter = Highlighter(
        highlight_fields=["question"],
        skip_fields=["course", "section"]
    )
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        assert "course" not in h
        assert "section" not in h
        assert "question" in h


def test_highlighter_pass_through_non_highlighted_fields(text_fields, sample_docs):
    """Test that non-highlighted, non-skipped fields pass through."""
    index = Index(text_fields, keyword_fields=["course"])
    index.fit(sample_docs)

    highlighter = Highlighter(
        highlight_fields=["question"],
        skip_fields=["course"]
    )
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    for h in highlighted:
        # question is highlighted
        assert "question" in h
        assert isinstance(h["question"], dict)
        # section passes through
        assert "section" in h
        assert isinstance(h["section"], str)
        # course is skipped
        assert "course" not in h


def test_highlighter_natural_language_query(text_fields, sample_docs):
    """Test highlighting with natural language queries."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])
    results = index.search("course join")
    highlighted = highlighter.highlight("I just discovered the course, can I still join?", results)

    assert len(highlighted) > 0
    # Should find the document about joining
    found_join = any(
        "join" in str(h.get("question", "")).lower() or "join" in str(h.get("text", "")).lower()
        for h in highlighted
    )
    assert found_join


def test_highlighter_with_appendable_index(text_fields, sample_docs):
    """Test that Highlighter works with AppendableIndex."""
    index = AppendableIndex(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    assert len(highlighted) > 0
    assert "question" in highlighted[0]


def test_highlighter_max_matches_limit(text_fields, sample_docs):
    """Test that max_matches limits the number of returned snippets."""
    docs = [
        {
            "question": "Python Python Python Python Python Python Python Python Python Python",
            "text": "Python is great",
        }
    ]

    index = Index(text_fields)
    index.fit(docs)

    highlighter = Highlighter(highlight_fields=["question"], max_matches=2)
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    # Should return at most 2 matches
    assert len(highlighted[0]["question"]["matches"]) <= 2
    # But total_matches should reflect all occurrences
    assert highlighted[0]["question"]["total_matches"] >= 2


def test_highlighter_total_matches_count(text_fields, sample_docs):
    """Test that total_matches reflects all occurrences."""
    docs = [
        {
            "question": "Python is great and Python is easy",
            "text": "Python programming",
        }
    ]

    index = Index(text_fields)
    index.fit(docs)

    highlighter = Highlighter(highlight_fields=["question"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    # Should count both occurrences
    assert highlighted[0]["question"]["total_matches"] == 2


def test_highlighter_empty_query(text_fields, sample_docs):
    """Test highlighting with empty query (only stop words)."""
    from minsearch import Tokenizer, AppendableIndex

    tokenizer = Tokenizer(stop_words='english')
    index = AppendableIndex(text_fields=text_fields, tokenizer=tokenizer)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question", "text"], tokenizer=tokenizer)
    results = index.search("python")

    # Query with only stop words
    highlighted = highlighter.highlight("the and is", results)

    # Should return empty matches structure
    for h in highlighted:
        # Fields should have consistent structure
        assert "question" in h
        assert isinstance(h["question"], dict)
        assert h["question"]["matches"] == []
        assert h["question"]["total_matches"] == 0


def test_highlighter_case_preservation(text_fields, sample_docs):
    """Test that highlighting preserves original case."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question"])
    results = index.search("PYTHON")  # Uppercase query

    highlighted = highlighter.highlight("PYTHON", results)

    for h in highlighted:
        if h["question"]["matches"]:
            # Should preserve original case from document
            assert "**Python**" in h["question"]["matches"][0]


def test_highlighter_no_matches(text_fields, sample_docs):
    """Test highlighting when query doesn't match."""
    docs = [{"text": "This is about Python programming"}]

    index = Index(text_fields=["text"])
    index.fit(docs)

    highlighter = Highlighter(highlight_fields=["text"])
    results = index.search("python")
    highlighted = highlighter.highlight("nonexistent term", results)

    assert highlighted[0]["text"]["matches"] == []
    assert highlighted[0]["text"]["total_matches"] == 0


def test_highlighter_with_filters(text_fields, sample_docs):
    """Test highlighting works with filtered search results."""
    index = Index(text_fields, keyword_fields=["course"])
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])
    results = index.search("python", filter_dict={"course": "python-basics"})
    highlighted = highlighter.highlight("python", results)

    assert len(highlighted) > 0
    # All original results should be from python-basics
    # (but course is pass-through since we didn't skip it)
    for h in highlighted:
        # section should pass through
        assert "section" in h


def test_highlighter_snippet_size():
    """Test that snippet_size affects match length."""
    docs = [{"text": "Python " * 100 + "machine learning" + " is great " * 100}]

    index = Index(text_fields=["text"])
    index.fit(docs)

    highlighter = Highlighter(highlight_fields=["text"], snippet_size=50)
    results = index.search("machine learning")
    highlighted = highlighter.highlight("machine learning", results)

    if highlighted[0]["text"]["matches"]:
        snippet = highlighted[0]["text"]["matches"][0]
        # Should be reasonably short
        assert len(snippet) < 200


def test_highlighter_with_boost(text_fields, sample_docs):
    """Test that highlighting works with boosted search."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])

    # Search with boost
    results = index.search("python", boost_dict={"question": 2.0})
    highlighted = highlighter.highlight("python", results)

    assert len(highlighted) > 0
    assert "question" in highlighted[0]


def test_highlighter_multiterm_query(text_fields, sample_docs):
    """Test highlighting with multi-term queries."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])
    results = index.search("python machine learning")
    highlighted = highlighter.highlight("python machine learning", results)

    # Should find highlights for multiple terms
    for h in highlighted:
        if h["question"]["matches"]:
            # At least one term should be highlighted
            assert "**" in h["question"]["matches"][0]


def test_highlighter_custom_stop_words():
    """Test Highlighter with custom stop words via tokenizer."""
    from minsearch.tokenizer import Tokenizer

    docs = [{"text": "The quick brown fox jumps over the lazy dog"}]

    index = Index(text_fields=["text"])
    index.fit(docs)

    # Use custom stop words (including "quick")
    tokenizer = Tokenizer(stop_words={"quick", "brown", "the", "over", "lazy"})
    highlighter = Highlighter(
        highlight_fields=["text"],
        tokenizer=tokenizer
    )
    results = index.search("quick brown")
    highlighted = highlighter.highlight("quick brown", results)

    # Should not highlight anything since all terms are stop words
    assert highlighted[0]["text"]["matches"] == []
    assert highlighted[0]["text"]["total_matches"] == 0


def test_highlighter_multiple_fields_with_different_results(text_fields, sample_docs):
    """Test highlighting multiple fields from the same document."""
    docs = [
        {
            "question": "Python programming tutorial",
            "text": "Learn Python programming in this comprehensive tutorial. Python is versatile.",
            "category": "Programming"
        }
    ]

    index = Index(text_fields=["question", "text", "category"])
    index.fit(docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    h = highlighted[0]
    # Both fields should have matches
    assert len(h["question"]["matches"]) > 0
    assert h["question"]["total_matches"] >= 1
    assert len(h["text"]["matches"]) > 0
    assert h["text"]["total_matches"] >= 2
    # category should pass through
    assert h["category"] == "Programming"


def test_highlighter_long_natural_language_query():
    """Test with a complex natural language query."""
    docs = [
        {
            "question": "Late enrollment policy",
            "text": "Students can join the course late. There's no penalty for late registration. You'll get full access to all materials.",
        },
        {
            "question": "Course prerequisites",
            "text": "This course requires basic programming knowledge. No prior experience with Python is needed.",
        },
    ]

    index = Index(text_fields=["question", "text"])
    index.fit(docs)

    highlighter = Highlighter(highlight_fields=["question", "text"])
    results = index.search("late")
    highlighted = highlighter.highlight("I just found this course, is it too late to enroll?", results)

    # Should find relevant content about late enrollment
    assert len(highlighted) > 0
    found_late = any(
        "late" in str(h.get("question", "")).lower() or "late" in str(h.get("text", "")).lower()
        for h in highlighted
    )
    assert found_late


def test_highlighter_returns_list_of_dicts(text_fields, sample_docs):
    """Test that highlight returns a list of dicts."""
    index = Index(text_fields)
    index.fit(sample_docs)

    highlighter = Highlighter(highlight_fields=["question"])
    results = index.search("python")
    highlighted = highlighter.highlight("python", results)

    assert isinstance(highlighted, list)
    for h in highlighted:
        assert isinstance(h, dict)


def test_highlighter_natural_language_query_long():
    """Test highlighting with a long natural language query."""
    from minsearch import Tokenizer, AppendableIndex

    tokenizer = Tokenizer(stop_words='english')
    docs = [
        {
            "question": "I just discovered the course, can I still join?",
            "text": "Yes, you can still join the course even after discovering it late. You will have access to all course materials from the start.",
            "section": "Enrollment",
        },
        {
            "question": "Late enrollment policy for courses",
            "text": "Students can enroll late. There is no penalty for late registration. Full access to materials is guaranteed.",
            "section": "Enrollment",
        },
        {
            "question": "Course prerequisites and requirements",
            "text": "This course requires basic Python knowledge. No prior experience with data science is needed.",
            "section": "Requirements",
        },
    ]

    index = AppendableIndex(text_fields=["question", "text", "section"], tokenizer=tokenizer)
    index.fit(docs)

    highlighter = Highlighter(highlight_fields=["question", "text"], tokenizer=tokenizer)

    # Natural language query
    query = "I just discovered the course, can I still join?"
    results = index.search(query)
    highlighted = highlighter.highlight(query, results)

    # Verify query extraction
    extracted_terms = highlighter._extract_query_terms(query)
    # Stop words removed: "I", "the"
    assert "just" in extracted_terms
    assert "discovered" in extracted_terms
    assert "course" in extracted_terms
    assert "can" in extracted_terms
    assert "still" in extracted_terms
    assert "join" in extracted_terms
    # Stop words NOT in extracted terms
    assert "i" not in extracted_terms
    assert "the" not in extracted_terms

    # First result should be the exact match
    first_result = highlighted[0]
    assert "question" in first_result
    assert "text" in first_result
    assert "section" in first_result
    assert first_result["section"] == "Enrollment"

    # Check question field has matches
    assert first_result["question"]["total_matches"] == 6
    assert len(first_result["question"]["matches"]) > 0
    # All terms should be highlighted
    assert "**just**" in first_result["question"]["matches"][0]
    assert "**discovered**" in first_result["question"]["matches"][0]
    assert "**course**" in first_result["question"]["matches"][0]
    assert "**can**" in first_result["question"]["matches"][0]
    assert "**still**" in first_result["question"]["matches"][0]
    assert "**join**" in first_result["question"]["matches"][0]

    # Check text field has matches
    assert first_result["text"]["total_matches"] >= 4
    assert len(first_result["text"]["matches"]) > 0
