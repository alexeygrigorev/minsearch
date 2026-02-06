import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from datetime import date, datetime


# Operator mapping for range filters
OPERATORS = {
    '>=': lambda a, b: a >= b,
    '>': lambda a, b: a > b,
    '<=': lambda a, b: a <= b,
    '<': lambda a, b: a < b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
}


class Index:
    """
    A simple search index using TF-IDF and cosine similarity for text fields,
    exact matching for keyword fields, and range filters for numeric and date fields.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        numeric_fields (list): List of numeric field names to index.
        date_fields (list): List of date field names to index.
        vectorizers (dict): Dictionary of TfidfVectorizer instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        numeric_df (pd.DataFrame): DataFrame containing numeric field data.
        date_df (pd.DataFrame): DataFrame containing date field data.
        text_matrices (dict): Dictionary of TF-IDF matrices for each text field.
        docs (list): List of documents indexed.
    """

    def __init__(self, text_fields, keyword_fields=None, numeric_fields=None, date_fields=None, vectorizer_params=None):
        """
        Initializes the Index with specified text, keyword, numeric, and date fields.

        Args:
            text_fields (list): List of text field names to index.
            keyword_fields (list, optional): List of keyword field names to index. Defaults to empty list.
            numeric_fields (list, optional): List of numeric field names to index. Defaults to empty list.
            date_fields (list, optional): List of date field names to index. Defaults to empty list.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields if keyword_fields is not None else []
        self.numeric_fields = numeric_fields if numeric_fields is not None else []
        self.date_fields = date_fields if date_fields is not None else []
        if vectorizer_params is None:
            vectorizer_params = {}

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
        self.numeric_df = None
        self.date_df = None
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
        numeric_data = {field: [] for field in self.numeric_fields}
        date_data = {field: [] for field in self.date_fields}

        # Handle empty documents case
        if not docs:
            self.keyword_df = pd.DataFrame(keyword_data)
            self.numeric_df = pd.DataFrame(numeric_data)
            self.date_df = pd.DataFrame(date_data)
            return self

        for field in self.text_fields:
            texts = [doc.get(field, '') or '' for doc in docs]
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
                keyword_data[field].append(doc.get(field))
            for field in self.numeric_fields:
                numeric_data[field].append(doc.get(field))
            for field in self.date_fields:
                value = doc.get(field)
                # Convert date/datetime objects to pandas timestamps for comparison
                if isinstance(value, (date, datetime)):
                    date_data[field].append(pd.Timestamp(value))
                else:
                    date_data[field].append(value)

        self.keyword_df = pd.DataFrame(keyword_data)
        self.numeric_df = pd.DataFrame(numeric_data)
        self.date_df = pd.DataFrame(date_data)

        return self

    def search(self, query, filter_dict=None, boost_dict=None, num_results=10, output_ids=False):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of filters. Can include:
                - Keyword fields: {"field": "value"} for exact match
                - Numeric/date fields: {"field": [('>=', value), ('<', value)]} for range filters
                - Multiple conditions on the same field are combined with AND
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.
            output_ids (bool): If True, adds an '_id' field to each document containing its index. Defaults to False.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
                         If output_ids is True, each document will have an additional '_id' field.
        """
        if filter_dict is None:
            filter_dict = {}
        if boost_dict is None:
            boost_dict = {}

        if not self.docs:
            return []

        query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        # Apply filters
        for field, value in filter_dict.items():
            # Keyword field filters (exact match)
            if field in self.keyword_fields:
                if value is None:
                    mask = self.keyword_df[field].isna()
                else:
                    mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

            # Numeric field filters (exact match or range comparisons)
            elif field in self.numeric_fields:
                if value is None:
                    # Filter for None values
                    mask = self.numeric_df[field].isna()
                    scores = scores * mask.to_numpy()
                elif isinstance(value, list) and all(isinstance(v, tuple) and len(v) == 2 for v in value):
                    # Range filter: [('>=', 10), ('<', 20)]
                    mask = np.ones(len(self.docs), dtype=bool)
                    for op, op_value in value:
                        if op in OPERATORS and op_value is not None:
                            series_mask = OPERATORS[op](self.numeric_df[field], op_value)
                            mask = mask & series_mask.to_numpy()
                    scores = scores * mask
                else:
                    # Exact match
                    mask = self.numeric_df[field] == value
                    scores = scores * mask.to_numpy()

            # Date field filters (exact match or range comparisons)
            elif field in self.date_fields:
                if value is None:
                    # Filter for None values
                    mask = self.date_df[field].isna()
                    scores = scores * mask.to_numpy()
                elif isinstance(value, list) and all(isinstance(v, tuple) and len(v) == 2 for v in value):
                    # Range filter: [('>=', date), ('<', date)]
                    mask = np.ones(len(self.docs), dtype=bool)
                    for op, op_value in value:
                        if op in OPERATORS and op_value is not None:
                            # Convert date/datetime to pandas Timestamp for comparison
                            if isinstance(op_value, (date, datetime)):
                                op_value = pd.Timestamp(op_value)
                            series_mask = OPERATORS[op](self.date_df[field], op_value)
                            mask = mask & series_mask.to_numpy()
                    scores = scores * mask
                else:
                    # Exact match (convert date/datetime to Timestamp)
                    if isinstance(value, (date, datetime)):
                        value = pd.Timestamp(value)
                    mask = self.date_df[field] == value
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
        if output_ids:
            return [{**self.docs[i], '_id': int(i)} for i in top_indices]
        return [self.docs[i] for i in top_indices]