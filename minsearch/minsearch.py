import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


class Index:
    """
    A simple search index using TF-IDF and cosine similarity for text fields and exact matching for keyword fields.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        text_matrices (dict): Dictionary of TF-IDF matrices for each text field.
        docs (list): List of documents indexed.
    """

    def __init__(self, text_fields, keyword_fields, vectorizer_params={}):
        """
        Initializes the Index with specified text and keyword fields.

        Args:
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields

        # Set default vectorizer parameters to ensure we always have terms
        default_params = {
            'min_df': 1,  # Include terms that appear in at least 1 document
            'max_df': 1.0,  # Include terms that appear in all documents
            'token_pattern': r'(?u)\b\w\w+\b',  # Match words with at least 2 characters
            'stop_words': None  # Don't remove any stop words by default
        }
        # Update with user parameters, but ensure defaults are used if not specified
        vectorizer_params = {**default_params, **vectorizer_params}

        self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    def fit(self, docs):
        """
        Fits the index with the provided documents.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
        """
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        # Handle empty documents case
        if not docs:
            self.keyword_df = pd.DataFrame(keyword_data)
            return self

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]
            try:
                self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)
            except ValueError as e:
                if "no terms remain" in str(e) or "empty vocabulary" in str(e):
                    # If no terms remain, create a dummy matrix with a single term
                    dummy_text = "dummy_term"  # A term that won't be filtered out
                    self.text_matrices[field] = self.vectorizers[field].fit_transform([dummy_text])
                else:
                    raise

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        if not self.docs:
            return []
            
        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        # Get number of non-zero scores
        non_zero_mask = scores > 0
        non_zero_count = np.sum(non_zero_mask)
        
        if non_zero_count == 0:
            return []
            
        # Ensure num_results doesn't exceed the number of non-zero scores
        num_results = min(num_results, non_zero_count)
        
        # Get indices of non-zero scores
        non_zero_indices = np.where(non_zero_mask)[0]
        
        # Sort non-zero scores in descending order
        sorted_indices = non_zero_indices[np.argsort(-scores[non_zero_indices])]
        
        # Take top num_results
        top_indices = sorted_indices[:num_results]
        
        # Return corresponding documents
        return [self.docs[i] for i in top_indices]