import numpy as np
import pytest
from datetime import date, datetime
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
    
    def test_append_single_vector(self):
        """Test appending a single vector to the index."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create initial vectors and payload
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "category": "programming"},
            {"id": 2, "title": "Doc 2", "category": "data"},
            {"id": 3, "title": "Doc 3", "category": "ai"}
        ]
        
        # Create and fit index
        index = VectorSearch(keyword_fields=["category"])
        index.fit(vectors, payload)
        
        # Append a new vector
        new_vector = np.random.rand(10)
        new_doc = {"id": 4, "title": "Doc 4", "category": "programming"}
        index.append(new_vector, new_doc)
        
        # Verify the index now has 4 vectors
        assert index.vectors.shape[0] == 4
        assert len(index.docs) == 4
        assert len(index.keyword_df) == 4
        
        # Search and verify the new document is searchable
        query_vector = new_vector
        results = index.search(query_vector, num_results=1)
        assert len(results) == 1
        assert results[0]["id"] == 4
        assert results[0]["title"] == "Doc 4"
    
    def test_append_to_empty_index(self):
        """Test appending to an empty index (without fit)."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create empty index
        index = VectorSearch(keyword_fields=["category"])
        
        # Append vectors one by one
        for i in range(3):
            vector = np.random.rand(10)
            doc = {"id": i+1, "title": f"Doc {i+1}", "category": "test"}
            index.append(vector, doc)
        
        # Verify the index has 3 vectors
        assert index.vectors.shape[0] == 3
        assert len(index.docs) == 3
        assert len(index.keyword_df) == 3
        
        # Search should work
        query_vector = np.random.rand(10)
        results = index.search(query_vector, num_results=2)
        assert len(results) == 2
    
    def test_append_batch(self):
        """Test appending multiple vectors in batch."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create initial vectors and payload
        initial_vectors = np.random.rand(2, 10)
        initial_payload = [
            {"id": 1, "title": "Doc 1", "category": "programming"},
            {"id": 2, "title": "Doc 2", "category": "data"}
        ]
        
        # Create and fit index
        index = VectorSearch(keyword_fields=["category"])
        index.fit(initial_vectors, initial_payload)
        
        # Append a batch of new vectors
        new_vectors = np.random.rand(3, 10)
        new_payload = [
            {"id": 3, "title": "Doc 3", "category": "ai"},
            {"id": 4, "title": "Doc 4", "category": "programming"},
            {"id": 5, "title": "Doc 5", "category": "data"}
        ]
        index.append_batch(new_vectors, new_payload)
        
        # Verify the index now has 5 vectors
        assert index.vectors.shape[0] == 5
        assert len(index.docs) == 5
        assert len(index.keyword_df) == 5
        
        # Search and verify all documents are searchable
        query_vector = np.random.rand(10)
        results = index.search(query_vector, num_results=5)
        assert len(results) == 5
    
    def test_append_batch_to_empty_index(self):
        """Test appending batch to an empty index (without fit)."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create empty index
        index = VectorSearch(keyword_fields=["category"])
        
        # Append a batch of vectors
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "category": "programming"},
            {"id": 2, "title": "Doc 2", "category": "data"},
            {"id": 3, "title": "Doc 3", "category": "ai"}
        ]
        index.append_batch(vectors, payload)
        
        # Verify the index has 3 vectors
        assert index.vectors.shape[0] == 3
        assert len(index.docs) == 3
        assert len(index.keyword_df) == 3
        
        # Search should work
        query_vector = np.random.rand(10)
        results = index.search(query_vector, num_results=2)
        assert len(results) == 2
    
    def test_append_batch_mismatch(self):
        """Test error when vectors and payload have different lengths in append_batch."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1"},
            {"id": 2, "title": "Doc 2"}
        ]
        
        index = VectorSearch(keyword_fields=[])
        
        with pytest.raises(ValueError, match="Number of vectors must match number of payload documents"):
            index.append_batch(vectors, payload)
    
    def test_append_with_keyword_filtering(self):
        """Test that appending preserves keyword filtering functionality."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create initial index
        index = VectorSearch(keyword_fields=["category"])
        initial_vectors = np.random.rand(2, 10)
        initial_payload = [
            {"id": 1, "title": "Doc 1", "category": "programming"},
            {"id": 2, "title": "Doc 2", "category": "data"}
        ]
        index.fit(initial_vectors, initial_payload)
        
        # Append a new vector
        new_vector = np.random.rand(10)
        new_doc = {"id": 3, "title": "Doc 3", "category": "programming"}
        index.append(new_vector, new_doc)
        
        # Search with filter
        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={"category": "programming"}, num_results=5)
        
        # Should only return programming documents (ids 1 and 3)
        assert len(results) == 2
        assert all(doc["category"] == "programming" for doc in results)
        assert set(doc["id"] for doc in results) == {1, 3}
    
    def test_mixed_append_and_append_batch(self):
        """Test mixing append and append_batch operations."""
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create empty index
        index = VectorSearch(keyword_fields=["category"])
        
        # Append single vector
        index.append(np.random.rand(10), {"id": 1, "title": "Doc 1", "category": "a"})
        
        # Append batch
        index.append_batch(
            np.random.rand(2, 10),
            [
                {"id": 2, "title": "Doc 2", "category": "b"},
                {"id": 3, "title": "Doc 3", "category": "c"}
            ]
        )
        
        # Append another single vector
        index.append(np.random.rand(10), {"id": 4, "title": "Doc 4", "category": "a"})
        
        # Verify the index has 4 vectors
        assert index.vectors.shape[0] == 4
        assert len(index.docs) == 4
        
        # Search should work
        query_vector = np.random.rand(10)
        results = index.search(query_vector, num_results=4)
        assert len(results) == 4


class TestVectorSearchRangeFilters:
    """Tests for numeric and date range filters in VectorSearch."""

    def test_numeric_filter_greater_than_or_equal(self):
        """Test numeric filter with >= operator."""
        np.random.seed(42)
        vectors = np.random.rand(4, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "price": 100},
            {"id": 2, "title": "Doc 2", "price": 200},
            {"id": 3, "title": "Doc 3", "price": 150},
            {"id": 4, "title": "Doc 4", "price": 50},
        ]

        index = VectorSearch(numeric_fields=["price"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={"price": [('>=', 100)]})
        assert all(doc["price"] >= 100 for doc in results)

    def test_numeric_filter_range(self):
        """Test numeric filter with range."""
        np.random.seed(42)
        vectors = np.random.rand(4, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "price": 100},
            {"id": 2, "title": "Doc 2", "price": 200},
            {"id": 3, "title": "Doc 3", "price": 150},
            {"id": 4, "title": "Doc 4", "price": 50},
        ]

        index = VectorSearch(numeric_fields=["price"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={"price": [('>', 100), ('<', 200)]})
        assert all(doc["price"] > 100 and doc["price"] < 200 for doc in results)

    def test_date_filter_greater_than_or_equal(self):
        """Test date filter with >= operator."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "created_at": date(2024, 1, 15)},
            {"id": 2, "title": "Doc 2", "created_at": date(2024, 2, 20)},
            {"id": 3, "title": "Doc 3", "created_at": date(2024, 3, 10)},
        ]

        index = VectorSearch(date_fields=["created_at"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={"created_at": [('>=', date(2024, 2, 1))]})
        assert all(doc["created_at"] >= date(2024, 2, 1) for doc in results)

    def test_datetime_filter(self):
        """Test datetime filter with datetime objects."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "created_at": datetime(2024, 1, 15, 10, 30)},
            {"id": 2, "title": "Doc 2", "created_at": datetime(2024, 2, 20, 14, 45)},
            {"id": 3, "title": "Doc 3", "created_at": datetime(2024, 3, 10, 8, 0)},
        ]

        index = VectorSearch(date_fields=["created_at"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={
            "created_at": [('>=', datetime(2024, 2, 1, 0, 0))]
        })
        assert all(doc["created_at"] >= datetime(2024, 2, 1, 0, 0) for doc in results)

    def test_combined_keyword_and_numeric_filters(self):
        """Test combining keyword and numeric filters."""
        np.random.seed(42)
        vectors = np.random.rand(4, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "price": 100, "category": "A"},
            {"id": 2, "title": "Doc 2", "price": 200, "category": "B"},
            {"id": 3, "title": "Doc 3", "price": 150, "category": "A"},
            {"id": 4, "title": "Doc 4", "price": 50, "category": "B"},
        ]

        index = VectorSearch(keyword_fields=["category"], numeric_fields=["price"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={
            "category": "A",
            "price": [('>=', 100)]
        })
        assert all(doc["category"] == "A" and doc["price"] >= 100 for doc in results)

    def test_append_with_numeric_field(self):
        """Test that appending preserves numeric field functionality."""
        np.random.seed(42)
        index = VectorSearch(numeric_fields=["price"])

        # Append initial vectors
        index.append(np.random.rand(10), {"id": 1, "title": "Doc 1", "price": 100})
        index.append(np.random.rand(10), {"id": 2, "title": "Doc 2", "price": 200})

        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={"price": [('>=', 150)]})
        assert all(doc["price"] >= 150 for doc in results)

    def test_append_batch_with_date_field(self):
        """Test that append_batch preserves date field functionality."""
        np.random.seed(42)
        index = VectorSearch(date_fields=["created_at"])

        # Append batch
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "created_at": date(2024, 1, 15)},
            {"id": 2, "title": "Doc 2", "created_at": date(2024, 2, 20)},
            {"id": 3, "title": "Doc 3", "created_at": date(2024, 3, 10)},
        ]
        index.append_batch(vectors, payload)

        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={
            "created_at": [('>=', date(2024, 2, 1))]
        })
        assert all(doc["created_at"] >= date(2024, 2, 1) for doc in results)

    def test_all_numeric_operators(self):
        """Test all supported numeric operators."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "value": 10},
            {"id": 2, "title": "Doc 2", "value": 20},
            {"id": 3, "title": "Doc 3", "value": 30},
        ]

        index = VectorSearch(numeric_fields=["value"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)

        # Test >=
        results = index.search(query_vector, filter_dict={"value": [('>=', 20)]})
        assert all(doc["value"] >= 20 for doc in results)

        # Test >
        results = index.search(query_vector, filter_dict={"value": [('>', 20)]})
        assert all(doc["value"] > 20 for doc in results)

        # Test <=
        results = index.search(query_vector, filter_dict={"value": [('<=', 20)]})
        assert all(doc["value"] <= 20 for doc in results)

        # Test <
        results = index.search(query_vector, filter_dict={"value": [('<', 20)]})
        assert all(doc["value"] < 20 for doc in results)

        # Test ==
        results = index.search(query_vector, filter_dict={"value": [('==', 20)]})
        assert all(doc["value"] == 20 for doc in results)

        # Test !=
        results = index.search(query_vector, filter_dict={"value": [('!=', 20)]})
        assert all(doc["value"] != 20 for doc in results)

    def test_numeric_exact_match_equality(self):
        """Test numeric field exact match with simple value (not list)."""
        np.random.seed(42)
        vectors = np.random.rand(4, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "price": 100},
            {"id": 2, "title": "Doc 2", "price": 200},
            {"id": 3, "title": "Doc 3", "price": 150},
            {"id": 4, "title": "Doc 4", "price": 50},
        ]

        index = VectorSearch(numeric_fields=["price"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # Exact match: price == 150
        results = index.search(query_vector, filter_dict={"price": 150})
        assert all(doc["price"] == 150 for doc in results)

    def test_date_exact_match_equality(self):
        """Test date field exact match with simple value (not list)."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "created_at": date(2024, 1, 15)},
            {"id": 2, "title": "Doc 2", "created_at": date(2024, 2, 20)},
            {"id": 3, "title": "Doc 3", "created_at": date(2024, 3, 10)},
        ]

        index = VectorSearch(date_fields=["created_at"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # Exact match: created_at == date(2024, 2, 20)
        results = index.search(query_vector, filter_dict={"created_at": date(2024, 2, 20)})
        assert all(doc["created_at"] == date(2024, 2, 20) for doc in results)

    def test_numeric_none_values(self):
        """Test numeric field with None values."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "price": 100},
            {"id": 2, "title": "Doc 2", "price": None},
            {"id": 3, "title": "Doc 3", "price": 200},
        ]

        index = VectorSearch(numeric_fields=["price"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # price == None
        results = index.search(query_vector, filter_dict={"price": None})
        assert all(doc["price"] is None for doc in results)

    def test_date_none_values(self):
        """Test date field with None values."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "created_at": date(2024, 1, 15)},
            {"id": 2, "title": "Doc 2", "created_at": None},
            {"id": 3, "title": "Doc 3", "created_at": date(2024, 2, 20)},
        ]

        index = VectorSearch(date_fields=["created_at"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # created_at == None
        results = index.search(query_vector, filter_dict={"created_at": None})
        assert all(doc["created_at"] is None for doc in results)

    def test_multiple_numeric_fields(self):
        """Test filtering on multiple numeric fields simultaneously."""
        np.random.seed(42)
        vectors = np.random.rand(4, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "price": 100, "rating": 4.5},
            {"id": 2, "title": "Doc 2", "price": 200, "rating": 3.8},
            {"id": 3, "title": "Doc 3", "price": 150, "rating": 4.2},
            {"id": 4, "title": "Doc 4", "price": 50, "rating": 4.8},
        ]

        index = VectorSearch(numeric_fields=["price", "rating"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # price >= 100 AND rating >= 4.0
        results = index.search(query_vector, filter_dict={
            "price": [('>=', 100)],
            "rating": [('>=', 4.0)]
        })
        assert all(doc["price"] >= 100 and doc["rating"] >= 4.0 for doc in results)

    def test_zero_values_in_numeric_fields(self):
        """Test that zero values are handled correctly in numeric fields."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "value": 0},
            {"id": 2, "title": "Doc 2", "value": 10},
            {"id": 3, "title": "Doc 3", "value": -5},
        ]

        index = VectorSearch(numeric_fields=["value"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # value == 0
        results = index.search(query_vector, filter_dict={"value": 0})
        assert all(doc["value"] == 0 for doc in results)

        # value >= 0
        results = index.search(query_vector, filter_dict={"value": [('>=', 0)]})
        assert all(doc["value"] >= 0 for doc in results)

    def test_combined_equality_and_range_filters(self):
        """Test mixing equality (simple value) and range (list) filters."""
        np.random.seed(42)
        vectors = np.random.rand(4, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "price": 100, "category": "A"},
            {"id": 2, "title": "Doc 2", "price": 200, "category": "B"},
            {"id": 3, "title": "Doc 3", "price": 150, "category": "A"},
            {"id": 4, "title": "Doc 4", "price": 50, "category": "A"},
        ]

        index = VectorSearch(keyword_fields=["category"], numeric_fields=["price"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # category == "A" (equality) AND price >= 100 (range)
        results = index.search(query_vector, filter_dict={
            "category": "A",
            "price": [('>=', 100)]
        })
        assert all(doc["category"] == "A" and doc["price"] >= 100 for doc in results)

    def test_negative_numeric_values(self):
        """Test that negative values work correctly in numeric fields."""
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"id": 1, "title": "Doc 1", "value": -10},
            {"id": 2, "title": "Doc 2", "value": -5},
            {"id": 3, "title": "Doc 3", "value": 0},
        ]

        index = VectorSearch(numeric_fields=["value"])
        index.fit(vectors, payload)

        query_vector = np.random.rand(10)
        # value >= -7
        results = index.search(query_vector, filter_dict={"value": [('>=', -7)]})
        assert all(doc["value"] >= -7 for doc in results)

        # -10 < value < 0
        results = index.search(query_vector, filter_dict={"value": [('>', -10), ('<', 0)]})
        assert all(doc["value"] > -10 and doc["value"] < 0 for doc in results)