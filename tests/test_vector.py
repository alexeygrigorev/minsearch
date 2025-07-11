import numpy as np
import pytest
from minsearch.vector import VectorSearch


class TestVectorSearch:
    
    def test_basic_vector_search(self):
        """Test basic vector search functionality."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create sample vectors and payload
        vectors = np.random.rand(5, 10)  # 5 docs, 10-dim vectors
        payload = [
            {"id": 1, "title": "Python Tutorial", "category": "programming"},
            {"id": 2, "title": "Data Science", "category": "data"},
            {"id": 3, "title": "Machine Learning", "category": "ai"},
            {"id": 4, "title": "Web Development", "category": "programming"},
            {"id": 5, "title": "Statistics", "category": "data"}
        ]
        
        # Create and fit index
        index = VectorSearch(keyword_fields=["category"])
        index.fit(vectors, payload)
        
        # Search with query vector
        query_vector = np.random.rand(10)
        results = index.search(query_vector, num_results=3)
        
        # Check results
        assert len(results) == 3
        assert all(isinstance(doc, dict) for doc in results)
        assert all("id" in doc for doc in results)
    
    def test_keyword_filtering(self):
        """Test keyword filtering functionality."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        vectors = np.random.rand(5, 10)
        payload = [
            {"id": 1, "title": "Python Tutorial", "category": "programming", "level": "beginner"},
            {"id": 2, "title": "Data Science", "category": "data", "level": "intermediate"},
            {"id": 3, "title": "Machine Learning", "category": "ai", "level": "advanced"},
            {"id": 4, "title": "Web Development", "category": "programming", "level": "intermediate"},
            {"id": 5, "title": "Statistics", "category": "data", "level": "beginner"}
        ]
        
        index = VectorSearch(keyword_fields=["category", "level"])
        index.fit(vectors, payload)
        
        # Search with filter
        query_vector = np.random.rand(10)
        filter_dict = {"category": "programming", "level": "beginner"}
        results = index.search(query_vector, filter_dict=filter_dict)
        
        # Check that only programming + beginner docs are returned
        assert len(results) == 1
        assert results[0]["category"] == "programming"
        assert results[0]["level"] == "beginner"
    
    def test_output_ids(self):
        """Test output_ids functionality."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1"},
            {"id": 2, "title": "Doc 2"},
            {"id": 3, "title": "Doc 3"}
        ]
        
        index = VectorSearch(keyword_fields=[])
        index.fit(vectors, payload)
        
        query_vector = np.random.rand(10)
        results = index.search(query_vector, output_ids=True, num_results=3)
        
        # Check that _id field is added
        assert all("_id" in doc for doc in results)
        assert all(isinstance(doc["_id"], int) for doc in results)
    
    def test_empty_results(self):
        """Test behavior when no results match filters."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "category": "programming"},
            {"id": 2, "title": "Doc 2", "category": "data"},
            {"id": 3, "title": "Doc 3", "category": "ai"}
        ]
        
        index = VectorSearch(keyword_fields=["category"])
        index.fit(vectors, payload)
        
        query_vector = np.random.rand(10)
        filter_dict = {"category": "nonexistent"}
        results = index.search(query_vector, filter_dict=filter_dict)
        
        assert len(results) == 0
    
    def test_vector_payload_mismatch(self):
        """Test error when vectors and payload have different lengths."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1"},
            {"id": 2, "title": "Doc 2"}
        ]
        
        index = VectorSearch(keyword_fields=[])
        
        with pytest.raises(ValueError, match="Number of vectors must match number of payload documents"):
            index.fit(vectors, payload)
    
    def test_sorting_correctness(self):
        """Test that search results are correctly sorted by similarity score."""

        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 0.0]
        ])
        
        payload = [
            {"id": 0, "title": "Most Similar"},
            {"id": 1, "title": "Second Most Similar"},
            {"id": 2, "title": "Third Most Similar"},
            {"id": 3, "title": "Fourth Most Similar"},
            {"id": 4, "title": "Least Similar"}
        ]
        
        index = VectorSearch(keyword_fields=[])
        index.fit(vectors, payload)
        
        # Query vector that should be most similar to vector 0
        query_vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        
        results = index.search(query_vector, num_results=5)
        
        # Check that results are sorted by similarity (descending)
        assert len(results) == 5
        
        # The query vector is identical to vector 0, so it should be first
        assert results[0]["id"] == 0
        assert results[0]["title"] == "Most Similar"
        
        # Vector 1 should be second (cosine similarity = 0.8)
        assert results[1]["id"] == 1
        assert results[1]["title"] == "Second Most Similar"
        
        # Vector 2 should be third (cosine similarity = 0.6)
        assert results[2]["id"] == 2
        assert results[2]["title"] == "Third Most Similar"
        
        # Vector 3 should be fourth (cosine similarity = 0.4)
        assert results[3]["id"] == 3
        assert results[3]["title"] == "Fourth Most Similar"
        
        # Vector 4 should be last (cosine similarity = 0.2)
        assert results[4]["id"] == 4
        assert results[4]["title"] == "Least Similar"