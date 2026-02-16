import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date, datetime

from sklearn.metrics.pairwise import cosine_similarity

from .filters import Filter, FieldData


class VectorSearch:
    """
    A vector search index using cosine similarity for vector search,
    exact matching for keyword fields, and range filters for numeric and date fields.

    Takes a 2D numpy array of vectors and a list of payload documents, providing efficient
    similarity search with keyword, numeric, and date filtering capabilities.
    """

    def __init__(self, keyword_fields=None, numeric_fields=None, date_fields=None):
        """
        Initialize the VectorSearch index.

        Args:
            keyword_fields (list, optional): List of keyword field names to index for exact matching. Defaults to empty list.
            numeric_fields (list, optional): List of numeric field names to index for range filters. Defaults to empty list.
            date_fields (list, optional): List of date field names to index for range filters. Defaults to empty list.
        """
        self.keyword_fields = keyword_fields or []
        self.numeric_fields = numeric_fields or []
        self.date_fields = date_fields or []
        self.vectors = None  # 2D numpy array of vectors
        self.keyword_df = None  # DataFrame containing keyword field data
        self.numeric_df = None  # DataFrame containing numeric field data
        self.date_df = None  # DataFrame containing date field data
        self.docs = []  # List of documents (payload)
        # Initialize the filter with empty data (will be updated on fit/append)
        self._filter = Filter(
            keyword=FieldData(fields=self.keyword_fields, data={}),
            numeric=FieldData(fields=self.numeric_fields, data={}),
            date=FieldData(fields=self.date_fields, data={}),
            num_docs=0,
        )
        
    def fit(self, vectors, payload):
        """
        Fits the index with the provided vectors and payload documents.

        Args:
            vectors (np.ndarray): 2D numpy array of shape (n_docs, vector_dimension).
            payload (list of dict): List of documents to use as payload. Each document is a dictionary.
        """
        if len(vectors) != len(payload):
            raise ValueError("Number of vectors must match number of payload documents")

        self.vectors = vectors
        self.docs = payload

        # Create keyword DataFrame
        keyword_data = {field: [] for field in self.keyword_fields}
        numeric_data = {field: [] for field in self.numeric_fields}
        date_data = {field: [] for field in self.date_fields}

        for doc in payload:
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

        # Initialize the filter
        self._filter = Filter(
            keyword=FieldData(fields=self.keyword_fields, data=self.keyword_df),
            numeric=FieldData(fields=self.numeric_fields, data=self.numeric_df),
            date=FieldData(fields=self.date_fields, data=self.date_df),
            num_docs=len(self.docs),
        )

        return self
    
    def append(self, vector, doc):
        """
        Appends a single vector and its corresponding document to the index.

        Args:
            vector (np.ndarray): 1D numpy array representing the vector to append.
            doc (dict): Document to append as payload.

        Returns:
            self: Returns the index instance for method chaining.
        """
        # Reshape vector to 2D if needed (ensure it's a row vector)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Initialize vectors if this is the first document
        if self.vectors is None:
            self.vectors = vector
            self.docs = [doc]

            # Initialize DataFrames
            keyword_data = {field: [doc.get(field)] for field in self.keyword_fields}
            self.keyword_df = pd.DataFrame(keyword_data)
            numeric_data = {field: [doc.get(field)] for field in self.numeric_fields}
            self.numeric_df = pd.DataFrame(numeric_data)
            date_data = {}
            for field in self.date_fields:
                value = doc.get(field)
                if isinstance(value, (date, datetime)):
                    date_data[field] = [pd.Timestamp(value)]
                else:
                    date_data[field] = [value]
            self.date_df = pd.DataFrame(date_data)
        else:
            # Append vector to existing vectors
            self.vectors = np.vstack([self.vectors, vector])
            self.docs.append(doc)

            # Append to DataFrames
            new_row = {field: doc.get(field) for field in self.keyword_fields}
            self.keyword_df = pd.concat([self.keyword_df, pd.DataFrame([new_row])], ignore_index=True)

            new_row = {field: doc.get(field) for field in self.numeric_fields}
            self.numeric_df = pd.concat([self.numeric_df, pd.DataFrame([new_row])], ignore_index=True)

            date_row = {}
            for field in self.date_fields:
                value = doc.get(field)
                if isinstance(value, (date, datetime)):
                    date_row[field] = pd.Timestamp(value)
                else:
                    date_row[field] = value
            self.date_df = pd.concat([self.date_df, pd.DataFrame([date_row])], ignore_index=True)

        # Update the filter
        self._filter.refresh(
            keyword_data=self.keyword_df,
            numeric_data=self.numeric_df,
            date_data=self.date_df,
            num_docs=len(self.docs),
        )

        return self
    
    def append_batch(self, vectors, payload):
        """
        Appends multiple vectors and their corresponding documents to the index.

        Args:
            vectors (np.ndarray): 2D numpy array of shape (n_docs, vector_dimension).
            payload (list of dict): List of documents to append as payload.

        Returns:
            self: Returns the index instance for method chaining.
        """
        if len(vectors) != len(payload):
            raise ValueError("Number of vectors must match number of payload documents")

        # Initialize vectors if this is the first batch
        if self.vectors is None:
            self.vectors = vectors
            self.docs = payload

            # Initialize DataFrames
            keyword_data = {field: [] for field in self.keyword_fields}
            numeric_data = {field: [] for field in self.numeric_fields}
            date_data = {field: [] for field in self.date_fields}

            for doc in payload:
                for field in self.keyword_fields:
                    keyword_data[field].append(doc.get(field))
                for field in self.numeric_fields:
                    numeric_data[field].append(doc.get(field))
                for field in self.date_fields:
                    value = doc.get(field)
                    if isinstance(value, (date, datetime)):
                        date_data[field].append(pd.Timestamp(value))
                    else:
                        date_data[field].append(value)

            self.keyword_df = pd.DataFrame(keyword_data)
            self.numeric_df = pd.DataFrame(numeric_data)
            self.date_df = pd.DataFrame(date_data)
        else:
            # Append vectors to existing vectors
            self.vectors = np.vstack([self.vectors, vectors])
            self.docs.extend(payload)

            # Append to DataFrames
            keyword_data = {field: [] for field in self.keyword_fields}
            numeric_data = {field: [] for field in self.numeric_fields}
            date_data = {field: [] for field in self.date_fields}

            for doc in payload:
                for field in self.keyword_fields:
                    keyword_data[field].append(doc.get(field))
                for field in self.numeric_fields:
                    numeric_data[field].append(doc.get(field))
                for field in self.date_fields:
                    value = doc.get(field)
                    if isinstance(value, (date, datetime)):
                        date_data[field].append(pd.Timestamp(value))
                    else:
                        date_data[field].append(value)

            self.keyword_df = pd.concat([self.keyword_df, pd.DataFrame(keyword_data)], ignore_index=True)
            self.numeric_df = pd.concat([self.numeric_df, pd.DataFrame(numeric_data)], ignore_index=True)
            self.date_df = pd.concat([self.date_df, pd.DataFrame(date_data)], ignore_index=True)

        # Update the filter
        self._filter.refresh(
            keyword_data=self.keyword_df,
            numeric_data=self.numeric_df,
            date_data=self.date_df,
            num_docs=len(self.docs),
        )

        return self
        
    def search(self, query_vector, filter_dict=None, num_results=10, output_ids=False):
        """
        Searches the index with the given query vector and filters.

        Args:
            query_vector (np.ndarray): 1D numpy array of shape (vector_dimension,) for the query.
            filter_dict (dict): Dictionary of filters. Can include:
                - Keyword fields: {"field": "value"} for exact match
                - Numeric/date fields: {"field": [('>=', value), ('<', value)]} for range filters
                - Multiple conditions on the same field are combined with AND
            num_results (int): The number of top results to return. Defaults to 10.
            output_ids (bool): If True, adds an '_id' field to each document containing its index.
                             Defaults to False.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
                         If output_ids is True, each document will have an additional '_id' field.
        """
        if filter_dict is None:
            filter_dict = {}

        if not self.docs or self.vectors is None:
            return []

        # Reshape query vector for cosine similarity
        query_vector_2d = query_vector.reshape(1, -1)

        # Compute cosine similarity
        scores = cosine_similarity(query_vector_2d, self.vectors).flatten()

        # Apply filters using the Filter object
        filter_mask = self._filter.apply(filter_dict)
        scores = scores * filter_mask

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

    def save(self, path: str | Path) -> None:
        """
        Save the index to a file using pickle.

        Args:
            path: File path to save the index to.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "VectorSearch":
        """
        Load an index from a file.

        Args:
            path: File path to load the index from.

        Returns:
            The loaded VectorSearch instance.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)