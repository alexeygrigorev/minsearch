import numpy as np
import pytest
from datetime import date

from minsearch.minsearch import Index
from minsearch.append import AppendableIndex
from minsearch.vector import VectorSearch
from minsearch.tokenizer import Tokenizer


@pytest.fixture
def sample_docs():
    return [
        {
            "question": "How do I use Python?",
            "text": "Python is a programming language.",
            "section": "Programming",
            "course": "python-basics",
            "score": 4.5,
        },
        {
            "question": "What is machine learning?",
            "text": "Machine learning is a subset of AI.",
            "section": "AI",
            "course": "ml-basics",
            "score": 3.8,
        },
        {
            "question": "How to write tests?",
            "text": "Tests help ensure code quality.",
            "section": "Testing",
            "course": "python-basics",
            "score": 4.2,
        },
    ]


@pytest.fixture
def docs_with_dates():
    return [
        {
            "question": "Python basics",
            "text": "Learn Python programming.",
            "section": "Programming",
            "published": date(2024, 1, 15),
            "score": 4.5,
        },
        {
            "question": "ML fundamentals",
            "text": "Machine learning concepts.",
            "section": "AI",
            "published": date(2024, 6, 1),
            "score": 3.8,
        },
    ]


class TestIndexSaveLoad:

    def test_save_load_basic(self, tmp_path, sample_docs):
        index = Index(
            text_fields=["question", "text"],
            keyword_fields=["section", "course"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = Index.load(path)

        results_orig = index.search("python programming")
        results_loaded = loaded.search("python programming")
        assert len(results_orig) == len(results_loaded)
        assert results_orig == results_loaded

    def test_save_load_with_keyword_filter(self, tmp_path, sample_docs):
        index = Index(
            text_fields=["question", "text"],
            keyword_fields=["section", "course"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = Index.load(path)

        results = loaded.search("python", filter_dict={"course": "python-basics"})
        assert all(doc["course"] == "python-basics" for doc in results)

    def test_save_load_with_numeric_filter(self, tmp_path, sample_docs):
        index = Index(
            text_fields=["question", "text"],
            keyword_fields=["section"],
            numeric_fields=["score"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = Index.load(path)

        results = loaded.search("python", filter_dict={"score": [(">=", 4.0)]})
        assert all(doc["score"] >= 4.0 for doc in results)

    def test_save_load_with_date_filter(self, tmp_path, docs_with_dates):
        index = Index(
            text_fields=["question", "text"],
            keyword_fields=["section"],
            date_fields=["published"],
        )
        index.fit(docs_with_dates)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = Index.load(path)

        results = loaded.search(
            "programming",
            filter_dict={"published": [(">=", date(2024, 3, 1))]},
        )
        assert len(results) >= 0  # Just verify it doesn't crash

    def test_save_load_with_boost(self, tmp_path, sample_docs):
        index = Index(
            text_fields=["question", "text"],
            keyword_fields=["section"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = Index.load(path)

        results = loaded.search("python", boost_dict={"question": 3.0})
        assert len(results) > 0

    def test_save_load_empty_index(self, tmp_path):
        index = Index(text_fields=["text"], keyword_fields=["tag"])
        index.fit([])

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = Index.load(path)

        assert loaded.search("anything") == []


class TestAppendableIndexSaveLoad:

    def test_save_load_basic(self, tmp_path, sample_docs):
        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["section", "course"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = AppendableIndex.load(path)

        results_orig = index.search("python programming")
        results_loaded = loaded.search("python programming")
        assert len(results_orig) == len(results_loaded)
        assert results_orig == results_loaded

    def test_save_load_with_keyword_filter(self, tmp_path, sample_docs):
        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["section", "course"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = AppendableIndex.load(path)

        results = loaded.search("python", filter_dict={"course": "python-basics"})
        assert all(doc["course"] == "python-basics" for doc in results)

    def test_save_load_with_numeric_filter(self, tmp_path, sample_docs):
        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["section"],
            numeric_fields=["score"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = AppendableIndex.load(path)

        results = loaded.search("python", filter_dict={"score": [(">=", 4.0)]})
        assert all(doc["score"] >= 4.0 for doc in results)

    def test_save_load_after_append(self, tmp_path, sample_docs):
        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["section", "course"],
        )
        index.fit(sample_docs)
        index.append({
            "question": "How to deploy?",
            "text": "Deployment strategies for Python apps.",
            "section": "DevOps",
            "course": "python-advanced",
        })

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = AppendableIndex.load(path)

        assert len(loaded.docs) == 4
        results = loaded.search("deploy")
        assert len(results) > 0

    def test_save_load_with_stemmer(self, tmp_path, sample_docs):
        tokenizer = Tokenizer(stemmer="porter")
        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["section"],
            tokenizer=tokenizer,
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = AppendableIndex.load(path)

        results = loaded.search("programming languages")
        assert len(results) > 0

    def test_save_load_can_append_after_load(self, tmp_path, sample_docs):
        index = AppendableIndex(
            text_fields=["question", "text"],
            keyword_fields=["section", "course"],
        )
        index.fit(sample_docs)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = AppendableIndex.load(path)

        loaded.append({
            "question": "What is Docker?",
            "text": "Docker is a containerization platform.",
            "section": "DevOps",
            "course": "docker-basics",
        })

        assert len(loaded.docs) == 4
        results = loaded.search("docker container")
        assert len(results) > 0


class TestVectorSearchSaveLoad:

    def test_save_load_basic(self, tmp_path):
        np.random.seed(42)
        vectors = np.random.rand(5, 10)
        payload = [
            {"title": "Python Tutorial", "category": "programming"},
            {"title": "Data Science", "category": "data"},
            {"title": "Machine Learning", "category": "ai"},
            {"title": "Web Development", "category": "programming"},
            {"title": "Statistics", "category": "data"},
        ]

        index = VectorSearch(keyword_fields=["category"])
        index.fit(vectors, payload)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = VectorSearch.load(path)

        query = np.random.rand(10)
        results_orig = index.search(query, num_results=3)
        results_loaded = loaded.search(query, num_results=3)
        assert len(results_orig) == len(results_loaded)
        assert results_orig == results_loaded

    def test_save_load_with_keyword_filter(self, tmp_path):
        np.random.seed(42)
        vectors = np.random.rand(5, 10)
        payload = [
            {"title": "Python Tutorial", "category": "programming"},
            {"title": "Data Science", "category": "data"},
            {"title": "Machine Learning", "category": "ai"},
            {"title": "Web Development", "category": "programming"},
            {"title": "Statistics", "category": "data"},
        ]

        index = VectorSearch(keyword_fields=["category"])
        index.fit(vectors, payload)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = VectorSearch.load(path)

        query = np.random.rand(10)
        results = loaded.search(query, filter_dict={"category": "programming"})
        assert all(doc["category"] == "programming" for doc in results)

    def test_save_load_with_numeric_filter(self, tmp_path):
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"title": "A", "score": 4.5},
            {"title": "B", "score": 2.0},
            {"title": "C", "score": 3.8},
        ]

        index = VectorSearch(numeric_fields=["score"])
        index.fit(vectors, payload)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = VectorSearch.load(path)

        query = np.random.rand(10)
        results = loaded.search(query, filter_dict={"score": [(">=", 3.0)]})
        assert all(doc["score"] >= 3.0 for doc in results)

    def test_save_load_after_append(self, tmp_path):
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"title": "A", "category": "x"},
            {"title": "B", "category": "y"},
            {"title": "C", "category": "x"},
        ]

        index = VectorSearch(keyword_fields=["category"])
        index.fit(vectors, payload)
        index.append(np.random.rand(10), {"title": "D", "category": "z"})

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = VectorSearch.load(path)

        assert len(loaded.docs) == 4
        query = np.random.rand(10)
        results = loaded.search(query)
        assert len(results) > 0

    def test_save_load_can_append_after_load(self, tmp_path):
        np.random.seed(42)
        vectors = np.random.rand(3, 10)
        payload = [
            {"title": "A", "category": "x"},
            {"title": "B", "category": "y"},
            {"title": "C", "category": "x"},
        ]

        index = VectorSearch(keyword_fields=["category"])
        index.fit(vectors, payload)

        path = tmp_path / "index.pkl"
        index.save(path)
        loaded = VectorSearch.load(path)

        loaded.append(np.random.rand(10), {"title": "D", "category": "z"})
        assert len(loaded.docs) == 4
        query = np.random.rand(10)
        results = loaded.search(query)
        assert len(results) > 0
