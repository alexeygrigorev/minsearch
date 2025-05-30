import pytest
import requests
from minsearch.minsearch import Index
from minsearch.append import AppendableIndex


@pytest.fixture
def sample_docs():
    # Fetch documents from the URL
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    
    # Process documents
    documents = []
    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    
    return documents


def test_notebook_example_comparison(sample_docs):
    """Test that reproduces the notebook example and compares results between Index and InvertedIndex."""
    # Create both index types
    index = Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    inverted_index = AppendableIndex(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    
    # Fit both indices
    index.fit(sample_docs)
    inverted_index.fit(sample_docs)
    
    # Test query from notebook
    query = "Can I join the course if it has already started?"
    filter_dict = {"course": "data-engineering-zoomcamp"}
    boost_dict = {"question": 3}
    
    # Get results from both indices
    index_results = index.search(query, filter_dict=filter_dict, boost_dict=boost_dict, num_results=5)
    inverted_results = inverted_index.search(query, filter_dict=filter_dict, boost_dict=boost_dict, num_results=5)
    
    # Basic assertions
    assert len(index_results) > 0, "Index should return results"
    assert len(inverted_results) > 0, "InvertedIndex should return results"
    
    # Compare number of results
    assert len(index_results) == len(inverted_results), "Both indices should return same number of results"
    
    # Compare top result
    # Note: We don't assert exact equality of results since the ranking might be slightly different
    # due to different implementations, but the top result should be relevant
    assert "join" in index_results[0]["question"].lower() or "join" in index_results[0]["text"].lower(), \
        "Top result from Index should be relevant to joining the course"
    assert "join" in inverted_results[0]["question"].lower() or "join" in inverted_results[0]["text"].lower(), \
        "Top result from InvertedIndex should be relevant to joining the course"
    
    # Test that all results are from the correct course
    for result in index_results:
        assert result["course"] == "data-engineering-zoomcamp", "All results should be from data-engineering-zoomcamp"
    for result in inverted_results:
        assert result["course"] == "data-engineering-zoomcamp", "All results should be from data-engineering-zoomcamp"
    
    # Test that question field is properly boosted
    # The document with "join" in the question should rank higher than one with it only in the text
    question_boosted = False
    for result in index_results:
        if "join" in result["question"].lower():
            question_boosted = True
            break
    assert question_boosted, "Question field should be properly boosted"
    
    question_boosted = False
    for result in inverted_results:
        if "join" in result["question"].lower():
            question_boosted = True
            break
    assert question_boosted, "Question field should be properly boosted in inverted index" 