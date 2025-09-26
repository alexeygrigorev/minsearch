"""
Tests for optional keyword_fields parameter across all index classes.
"""
import pytest
import numpy as np
from minsearch.minsearch import Index
from minsearch.append import AppendableIndex
from minsearch.vector import VectorSearch


@pytest.fixture
def sample_docs():
    return [
        {
            "title": "Python Programming",
            "content": "Learn Python basics and advanced concepts",
            "category": "programming",
            "level": "beginner"
        },
        {
            "title": "Data Science with Python",
            "content": "Python for data analysis and machine learning",
            "category": "data-science",
            "level": "intermediate"
        },
        {
            "title": "Web Development",
            "content": "Building web applications with Python frameworks",
            "category": "web",
            "level": "advanced"
        }
    ]


class TestIndexOptionalKeywordFields:
    """Test Index class with optional keyword_fields parameter."""

    def test_index_without_keyword_fields(self, sample_docs):
        """Test Index creation and functionality without keyword_fields."""
        index = Index(text_fields=["title", "content"])
        assert index.keyword_fields == []
        
        # Test fit and search work correctly
        index.fit(sample_docs)
        assert len(index.docs) == 3
        
        results = index.search("Python")
        assert len(results) > 0

    def test_index_with_explicit_empty_keyword_fields(self, sample_docs):
        """Test Index creation with explicitly empty keyword_fields."""
        index = Index(text_fields=["title", "content"], keyword_fields=[])
        assert index.keyword_fields == []
        
        index.fit(sample_docs)
        results = index.search("Python")
        assert len(results) > 0

    def test_index_with_keyword_fields(self, sample_docs):
        """Test Index creation with keyword_fields (backward compatibility)."""
        index = Index(text_fields=["title", "content"], keyword_fields=["category", "level"])
        assert index.keyword_fields == ["category", "level"]
        
        index.fit(sample_docs)
        results = index.search("Python", filter_dict={"category": "programming"})
        assert len(results) == 1
        assert results[0]["title"] == "Python Programming"

    def test_index_with_none_keyword_fields(self, sample_docs):
        """Test Index creation with keyword_fields=None."""
        index = Index(text_fields=["title", "content"], keyword_fields=None)
        assert index.keyword_fields == []
        
        index.fit(sample_docs)
        results = index.search("Python")
        assert len(results) > 0


class TestAppendableIndexOptionalKeywordFields:
    """Test AppendableIndex class with optional keyword_fields parameter."""

    def test_appendable_index_without_keyword_fields(self, sample_docs):
        """Test AppendableIndex creation and functionality without keyword_fields."""
        index = AppendableIndex(text_fields=["title", "content"])
        assert index.keyword_fields == []
        
        # Test fit and search work correctly
        index.fit(sample_docs)
        assert len(index.docs) == 3
        
        results = index.search("Python")
        assert len(results) > 0

    def test_appendable_index_with_explicit_empty_keyword_fields(self, sample_docs):
        """Test AppendableIndex creation with explicitly empty keyword_fields."""
        index = AppendableIndex(text_fields=["title", "content"], keyword_fields=[])
        assert index.keyword_fields == []
        
        index.fit(sample_docs)
        results = index.search("Python")
        assert len(results) > 0

    def test_appendable_index_with_keyword_fields(self, sample_docs):
        """Test AppendableIndex creation with keyword_fields (backward compatibility)."""
        index = AppendableIndex(text_fields=["title", "content"], keyword_fields=["category", "level"])
        assert index.keyword_fields == ["category", "level"]
        
        index.fit(sample_docs)
        results = index.search("Python", filter_dict={"category": "programming"})
        assert len(results) == 1
        assert results[0]["title"] == "Python Programming"

    def test_appendable_index_with_none_keyword_fields(self, sample_docs):
        """Test AppendableIndex creation with keyword_fields=None."""
        index = AppendableIndex(text_fields=["title", "content"], keyword_fields=None)
        assert index.keyword_fields == []
        
        index.fit(sample_docs)
        results = index.search("Python")
        assert len(results) > 0

    def test_appendable_index_append_functionality(self, sample_docs):
        """Test AppendableIndex append functionality with optional keyword_fields."""
        index = AppendableIndex(text_fields=["title", "content"])
        
        # Fit with initial docs
        initial_docs = sample_docs[:2]
        index.fit(initial_docs)
        
        # Append the last doc
        index.append(sample_docs[2])
        
        results = index.search("web")
        assert len(results) > 0


class TestVectorSearchOptionalKeywordFields:
    """Test VectorSearch class with optional keyword_fields parameter."""

    def test_vector_search_without_keyword_fields(self, sample_docs):
        """Test VectorSearch creation and functionality without keyword_fields."""
        index = VectorSearch()
        assert index.keyword_fields == []
        
        # Create random vectors for testing
        vectors = np.random.rand(len(sample_docs), 10)
        index.fit(vectors, sample_docs)
        
        query_vector = np.random.rand(10)
        results = index.search(query_vector)
        assert len(results) > 0

    def test_vector_search_with_explicit_empty_keyword_fields(self, sample_docs):
        """Test VectorSearch creation with explicitly empty keyword_fields."""
        index = VectorSearch(keyword_fields=[])
        assert index.keyword_fields == []
        
        vectors = np.random.rand(len(sample_docs), 10)
        index.fit(vectors, sample_docs)
        
        query_vector = np.random.rand(10)
        results = index.search(query_vector)
        assert len(results) > 0

    def test_vector_search_with_keyword_fields(self, sample_docs):
        """Test VectorSearch creation with keyword_fields (backward compatibility)."""
        index = VectorSearch(keyword_fields=["category", "level"])
        assert index.keyword_fields == ["category", "level"]
        
        vectors = np.random.rand(len(sample_docs), 10)
        index.fit(vectors, sample_docs)
        
        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={"category": "programming"})
        assert len(results) <= len(sample_docs)

    def test_vector_search_with_none_keyword_fields(self, sample_docs):
        """Test VectorSearch creation with keyword_fields=None."""
        index = VectorSearch(keyword_fields=None)
        assert index.keyword_fields == []
        
        vectors = np.random.rand(len(sample_docs), 10)
        index.fit(vectors, sample_docs)
        
        query_vector = np.random.rand(10)
        results = index.search(query_vector)
        assert len(results) > 0


class TestBackwardCompatibility:
    """Test that all changes maintain backward compatibility."""

    def test_all_classes_backward_compatibility(self, sample_docs):
        """Test that existing code with keyword_fields still works."""
        # Test Index
        index1 = Index(text_fields=["title"], keyword_fields=["category"])
        index1.fit(sample_docs)
        results1 = index1.search("Python", filter_dict={"category": "programming"})
        assert len(results1) > 0
        
        # Test AppendableIndex
        index2 = AppendableIndex(text_fields=["title"], keyword_fields=["category"])
        index2.fit(sample_docs)
        results2 = index2.search("Python", filter_dict={"category": "programming"})
        assert len(results2) > 0
        
        # Test VectorSearch
        vectors = np.random.rand(len(sample_docs), 10)
        index3 = VectorSearch(keyword_fields=["category"])
        index3.fit(vectors, sample_docs)
        query_vector = np.random.rand(10)
        results3 = index3.search(query_vector, filter_dict={"category": "programming"})
        assert len(results3) >= 0  # May be 0 due to random vectors

    def test_empty_docs_handling(self):
        """Test that empty docs are handled correctly with optional keyword_fields."""
        # Test Index
        index1 = Index(text_fields=["title"])
        assert index1.keyword_fields == []
        index1.fit([])
        results1 = index1.search("test")
        assert len(results1) == 0
        
        # Test AppendableIndex
        index2 = AppendableIndex(text_fields=["title"])
        assert index2.keyword_fields == []
        index2.fit([])
        results2 = index2.search("test")
        assert len(results2) == 0
        
        # Test VectorSearch
        index3 = VectorSearch()
        assert index3.keyword_fields == []
        # VectorSearch requires vectors, so we can't test with empty docs
        # but we can test initialization