"""
Tests for handling documents with None values in keyword fields.
"""
import pytest
import numpy as np
from minsearch.minsearch import Index
from minsearch.append import AppendableIndex
from minsearch.vector import VectorSearch


@pytest.fixture
def docs_with_none():
    return [
        {
            'url': 'https://api.github.com/repos/pydantic/pydantic-ai/issues/2932',
            'user_login': 'DouweM',
            'assignee_login': None,
            'state': 'open',
            'body': '### Description\n\nWe could support `Mem0` through a new toolset with `save_memory` and `search_memory` tools, similar to these examples:\n- OpenAI Agents SDK @https://docs.mem0.ai/integrations/openai-agents-sdk#basic-integration-example\n- LangChain Tools: @https://docs.mem0.ai/integrations/langchain-tools\n    - I expect these to already work with Pydantic AI via @https://ai.pydantic.dev/toolsets/#langchain-tools\n\n\n\n### References\n\n_No response_'
        },
        {
            'url': 'https://api.github.com/repos/pydantic/pydantic-ai/issues/2931',
            'user_login': 'JohnDoe',
            'assignee_login': 'JaneSmith',
            'state': 'closed',
            'body': 'Some other tools description'
        },
        {
            'url': 'https://api.github.com/repos/pydantic/pydantic-ai/issues/2930',
            'user_login': 'AliceWonder',
            'assignee_login': None,
            'state': 'open',
            'body': 'Another tools with no assignee'
        }
    ]


class TestIndexWithNoneValues:
    """Test Index class with None values in keyword fields."""

    def test_index_with_none_values(self, docs_with_none):
        """Test Index can handle None values in keyword fields."""
        index = Index(
            text_fields=["body"],
            keyword_fields=["assignee_login", "state"]
        )
        
        index.fit(docs_with_none)
        assert len(index.docs) == 3
        
        results = index.search("tools")
        assert len(results) > 0

    def test_index_search_with_none_filter(self, docs_with_none):
        """Test Index search with None value in filter."""
        index = Index(
            text_fields=["body"],
            keyword_fields=["assignee_login", "state"]
        )
        
        index.fit(docs_with_none)
        
        results = index.search("tools", filter_dict={"assignee_login": None}, num_results=10)
        assert len(results) == 2
        for result in results:
            assert result['assignee_login'] is None


class TestAppendableIndexWithNoneValues:
    """Test AppendableIndex class with None values in keyword fields."""

    def test_appendable_index_with_none_values(self, docs_with_none):
        """Test AppendableIndex can handle None values in keyword fields."""
        index = AppendableIndex(
            text_fields=["body"],
            keyword_fields=["assignee_login", "state"]
        )
        
        index.fit(docs_with_none)
        assert len(index.docs) == 3
        
        results = index.search("tools")
        assert len(results) > 0

    def test_appendable_index_search_with_none_filter(self, docs_with_none):
        """Test AppendableIndex search with None value in filter."""
        index = AppendableIndex(
            text_fields=["body"],
            keyword_fields=["assignee_login", "state"]
        )
        
        index.fit(docs_with_none)
        
        results = index.search("tools", filter_dict={"assignee_login": None}, num_results=10)
        assert len(results) == 2
        for result in results:
            assert result['assignee_login'] is None


class TestVectorSearchWithNoneValues:
    """Test VectorSearch class with None values in keyword fields."""

    def test_vector_search_with_none_values(self, docs_with_none):
        """Test VectorSearch can handle None values in keyword fields."""
        index = VectorSearch(keyword_fields=["assignee_login", "state"])
        
        vectors = np.random.rand(len(docs_with_none), 10)
        index.fit(vectors, docs_with_none)
        
        query_vector = np.random.rand(10)
        results = index.search(query_vector)
        assert len(results) > 0

    def test_vector_search_with_none_filter(self, docs_with_none):
        """Test VectorSearch search with None value in filter."""
        index = VectorSearch(keyword_fields=["assignee_login", "state"])
        
        vectors = np.random.rand(len(docs_with_none), 10)
        index.fit(vectors, docs_with_none)
        
        query_vector = np.random.rand(10)
        results = index.search(query_vector, filter_dict={"assignee_login": None}, num_results=10)
        assert len(results) == 2
        for result in results:
            assert result['assignee_login'] is None
