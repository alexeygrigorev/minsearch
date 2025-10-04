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