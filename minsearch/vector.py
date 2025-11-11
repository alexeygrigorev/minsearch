import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


class VectorSearch:
    """
    A vector search index using cosine similarity for vector search and exact matching for keyword fields.
    
    Takes a 2D numpy array of vectors and a list of payload documents, providing efficient
    similarity search with keyword filtering and boosting capabilities.
    """
    
    def __init__(self, keyword_fields=None):
        """
        Initialize the VectorSearch index.
        
        Args:
            keyword_fields (list, optional): List of keyword field names to index for exact matching. Defaults to empty list.
        """
        self.keyword_fields = keyword_fields if keyword_fields is not None else []
        self.vectors = None  # 2D numpy array of vectors
        self.keyword_df = None  # DataFrame containing keyword field data
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
        for doc in payload:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field))
        self.keyword_df = pd.DataFrame(keyword_data)
        
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
            
            # Initialize keyword DataFrame
            keyword_data = {field: [doc.get(field)] for field in self.keyword_fields}
            self.keyword_df = pd.DataFrame(keyword_data)
        else:
            # Append vector to existing vectors
            self.vectors = np.vstack([self.vectors, vector])
            self.docs.append(doc)
            
            # Append to keyword DataFrame
            new_row = {field: doc.get(field) for field in self.keyword_fields}
            self.keyword_df = pd.concat([self.keyword_df, pd.DataFrame([new_row])], ignore_index=True)
        
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
            
            # Initialize keyword DataFrame
            keyword_data = {field: [] for field in self.keyword_fields}
            for doc in payload:
                for field in self.keyword_fields:
                    keyword_data[field].append(doc.get(field))
            self.keyword_df = pd.DataFrame(keyword_data)
        else:
            # Append vectors to existing vectors
            self.vectors = np.vstack([self.vectors, vectors])
            self.docs.extend(payload)
            
            # Append to keyword DataFrame
            keyword_data = {field: [] for field in self.keyword_fields}
            for doc in payload:
                for field in self.keyword_fields:
                    keyword_data[field].append(doc.get(field))
            new_df = pd.DataFrame(keyword_data)
            self.keyword_df = pd.concat([self.keyword_df, new_df], ignore_index=True)
        
        return self
        
    def search(self, query_vector, filter_dict=None, num_results=10, output_ids=False):
        """
        Searches the index with the given query vector and filters.
        
        Args:
            query_vector (np.ndarray): 1D numpy array of shape (vector_dimension,) for the query.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names 
                              and values are the values to filter by.
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
        
        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                if value is None:
                    mask = self.keyword_df[field].isna()
                else:
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
        if output_ids:
            return [{**self.docs[i], '_id': int(i)} for i in top_indices]
        return [self.docs[i] for i in top_indices] 