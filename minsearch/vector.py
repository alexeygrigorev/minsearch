import numpy as np
import pandas as pd
from datetime import date, datetime

from sklearn.metrics.pairwise import cosine_similarity


# Operator mapping for range filters
OPERATORS = {
    '>=': lambda a, b: a >= b,
    '>': lambda a, b: a > b,
    '<=': lambda a, b: a <= b,
    '<': lambda a, b: a < b,
    '==': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
}


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
        self.keyword_fields = keyword_fields if keyword_fields is not None else []
        self.numeric_fields = numeric_fields if numeric_fields is not None else []
        self.date_fields = date_fields if date_fields is not None else []
        self.vectors = None  # 2D numpy array of vectors
        self.keyword_df = None  # DataFrame containing keyword field data
        self.numeric_df = None  # DataFrame containing numeric field data
        self.date_df = None  # DataFrame containing date field data
        self.docs = []  # List of documents (payload)
        
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