import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Union, Callable
from datetime import date, datetime
import numpy as np
import pandas as pd

from .filters import Filter, FieldData
from .tokenizer import Tokenizer


class AppendableIndex:
    """
    An appendable search index using inverted index for text fields,
    exact matching for keyword fields, and range filters for numeric and date fields.

    Performance optimizations:
    - Caches tokenized documents to avoid re-tokenization during search
    - Uses sets for inverted index (no duplicates, O(1) doc frequency)
    - Pre-computes IDF values for all tokens

    Attributes:
        text_fields: List of text field names to index.
        keyword_fields: List of keyword field names to index.
        numeric_fields: List of numeric field names to index.
        date_fields: List of date field names to index.
        docs: List of documents indexed.
        inverted_index: Dict mapping field -> token -> set of doc IDs.
        doc_frequencies: Dict mapping field -> token -> document frequency.
        idf: Dict mapping field -> token -> pre-computed IDF value.
        doc_tokens: Dict mapping field -> doc_id -> list of tokens (cached).
        total_docs: Total number of documents in the index.
    """

    def __init__(
        self,
        text_fields: List[str],
        keyword_fields: Optional[List[str]] = None,
        numeric_fields: Optional[List[str]] = None,
        date_fields: Optional[List[str]] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize the AppendableIndex.

        Args:
            text_fields: List of text field names to index.
            keyword_fields: List of keyword field names to index.
            numeric_fields: List of numeric field names to index.
            date_fields: List of date field names to index.
            tokenizer: Tokenizer to use. If None, creates a default one.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields if keyword_fields is not None else []
        self.numeric_fields = numeric_fields if numeric_fields is not None else []
        self.date_fields = date_fields if date_fields is not None else []

        # Initialize data structures
        self.docs = []
        self.total_docs = 0

        # Use sets for inverted index (no duplicates, O(1) insert)
        self.inverted_index: Dict[str, Dict[str, Set[int]]] = {
            field: defaultdict(set) for field in text_fields
        }

        # Document frequencies (now just counting unique docs)
        self.doc_frequencies: Dict[str, Dict[str, int]] = {
            field: defaultdict(int) for field in text_fields
        }

        # Pre-computed IDF values (computed after fit)
        self.idf: Dict[str, Dict[str, float]] = {
            field: {} for field in text_fields
        }

        # Cached tokenized documents per field
        self.doc_tokens: Dict[str, Dict[int, List[str]]] = {
            field: {} for field in text_fields
        }

        # Pre-computed token counts per document (for faster scoring)
        self.doc_token_counts: Dict[str, Dict[int, Dict[str, int]]] = {
            field: {} for field in text_fields
        }

        # Keyword data storage
        self.keyword_data: Dict[str, List] = {
            field: [] for field in self.keyword_fields
        }

        # Numeric data storage
        self.numeric_data: Dict[str, List] = {
            field: [] for field in self.numeric_fields
        }

        # Date data storage
        self.date_data: Dict[str, List] = {
            field: [] for field in self.date_fields
        }

        # Vocabularies for validation
        self.vocabularies: Dict[str, Set[str]] = {
            field: set() for field in text_fields
        }

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer()  # No stop words by default

        # Initialize the filter with empty data (will be updated on fit/append)
        self._filter = Filter(
            keyword=FieldData(fields=self.keyword_fields, data={}),
            numeric=FieldData(fields=self.numeric_fields, data={}),
            date=FieldData(fields=self.date_fields, data={}),
            num_docs=0,
        )

    def _process_text(self, text: str) -> List[str]:
        """Process text into tokens using our custom tokenizer."""
        return self.tokenizer.tokenize(text)

    def _compute_idf(self, field: str, doc_frequency: int) -> float:
        """Compute IDF for a token given its document frequency."""
        return math.log((self.total_docs + 1) / (doc_frequency + 1)) + 1

    def _update_inverted_index(self, doc_id: int, field: str, text: str) -> List[str]:
        """
        Update the inverted index for a given field and document.

        Returns the tokenized document for caching.
        """
        tokens = self._process_text(text)
        if not tokens:
            return tokens

        # Cache tokenized document
        self.doc_tokens[field][doc_id] = tokens

        # Pre-compute token counts for faster scoring
        token_counts = Counter(tokens)
        self.doc_token_counts[field][doc_id] = dict(token_counts)

        # Update inverted index with sets (no duplicates)
        for token in tokens:
            self.inverted_index[field][token].add(doc_id)

        # After all updates, recompute doc frequencies from actual set sizes
        for token in set(tokens):  # unique tokens in this doc
            self.doc_frequencies[field][token] = len(
                self.inverted_index[field][token]
            )
            self.vocabularies[field].add(token)

        return tokens

    def _finalize_index(self):
        """
        Finalize the index after fit/append operations.
        Pre-computes IDF values for all tokens and updates the filter.
        """
        for field in self.text_fields:
            self.idf[field] = {}
            for token, df in self.doc_frequencies[field].items():
                self.idf[field][token] = self._compute_idf(field, df)

    def _get_matching_documents(self, field: str, tokens: List[str]) -> Set[int]:
        """Get set of document IDs that match any of the given tokens."""
        matching_docs = set()
        for token in tokens:
            if token in self.inverted_index[field]:
                matching_docs.update(self.inverted_index[field][token])
        return matching_docs

    def _create_query_vector(
        self, field: str, query_tokens: List[str]
    ) -> Optional[tuple[np.ndarray, List[str]]]:
        """
        Create and normalize TF-IDF vector for the query.

        Returns:
            (query_vector, field_tokens) or (None, None) if no matches
        """
        # Filter to tokens that exist in our index
        field_tokens = [t for t in query_tokens if t in self.inverted_index[field]]
        if not field_tokens:
            return None, None

        # Count query tokens once (O(n) instead of O(n^2))
        query_token_counts = Counter(query_tokens)

        # Calculate query vector
        query_vector = np.zeros(len(field_tokens))
        for i, token in enumerate(field_tokens):
            # TF in query (sublinear TF scaling like sklearn)
            tf = query_token_counts[token]
            if tf > 0:
                tf = 1 + math.log(tf)
            # Use pre-computed IDF
            idf = self.idf[field].get(token, 1.0)
            query_vector[i] = tf * idf

        # L2 normalize
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        return query_vector, field_tokens

    def _calculate_document_score(
        self, field: str, doc_id: int, query_vector: np.ndarray, field_tokens: List[str]
    ) -> float:
        """
        Calculate cosine similarity between query and a document.

        Uses pre-computed token counts for efficiency.
        Only considers tokens that are in the query.
        """
        # Get pre-computed token counts for this document
        doc_token_counts = self.doc_token_counts[field].get(doc_id)
        if not doc_token_counts:
            return 0.0

        # Build document vector ONLY for query tokens, compute dot product and norm
        dot_product = 0.0
        doc_norm_sq = 0.0

        for i, token in enumerate(field_tokens):
            count = doc_token_counts.get(token)
            if count is not None:
                # Sublinear TF scaling
                tf = 1 + math.log(count)
                idf = self.idf[field].get(token, 1.0)
                tfidf = tf * idf
                dot_product += query_vector[i] * tfidf
                doc_norm_sq += tfidf * tfidf

        # Cosine similarity (only over query tokens)
        if doc_norm_sq == 0:
            return 0.0
        return dot_product / math.sqrt(doc_norm_sq)

    def _calculate_field_scores(
        self,
        field: str,
        query_vector: np.ndarray,
        field_tokens: List[str],
        matching_docs: Set[int],
    ) -> np.ndarray:
        """Calculate cosine similarity scores for all matching documents."""
        field_scores = np.zeros(len(self.docs))

        for doc_id in matching_docs:
            field_scores[doc_id] = self._calculate_document_score(
                field, doc_id, query_vector, field_tokens
            )

        return field_scores

    def fit(self, docs: List[Dict]) -> "AppendableIndex":
        """
        Fit the index with the provided documents.

        Args:
            docs: List of documents to index. Each document is a dictionary.

        Returns:
            self
        """
        self.docs = docs
        self.total_docs = len(docs)

        # Clear existing data
        for field in self.text_fields:
            self.inverted_index[field] = defaultdict(set)
            self.doc_frequencies[field] = defaultdict(int)
            self.doc_tokens[field] = {}
            self.doc_token_counts[field] = {}
            self.vocabularies[field] = set()

        for field in self.keyword_fields:
            self.keyword_data[field] = []

        for field in self.numeric_fields:
            self.numeric_data[field] = []

        for field in self.date_fields:
            self.date_data[field] = []

        # Process each document
        for doc_id, doc in enumerate(docs):
            # Update inverted index for text fields
            for field in self.text_fields:
                self._update_inverted_index(doc_id, field, doc.get(field, ""))

            # Collect keyword data
            for field in self.keyword_fields:
                self.keyword_data[field].append(doc.get(field))

            # Collect numeric data
            for field in self.numeric_fields:
                self.numeric_data[field].append(doc.get(field))

            # Collect date data
            for field in self.date_fields:
                value = doc.get(field)
                if isinstance(value, (date, datetime)):
                    self.date_data[field].append(pd.Timestamp(value))
                else:
                    self.date_data[field].append(value)

        # Validate vocabulary
        if self.docs:
            has_vocabulary = any(len(vocab) > 0 for vocab in self.vocabularies.values())
            if not has_vocabulary:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )

        # Pre-compute IDF values
        self._finalize_index()

        # Initialize the filter
        self._filter = Filter(
            keyword=FieldData(fields=self.keyword_fields, data=self.keyword_data),
            numeric=FieldData(fields=self.numeric_fields, data=self.numeric_data),
            date=FieldData(fields=self.date_fields, data=self.date_data),
            num_docs=len(self.docs),
        )

        return self

    def append(self, doc: Dict) -> "AppendableIndex":
        """
        Append a single document to the index.

        Note: This re-computes IDF values. For bulk additions, consider
        creating a new index or batch appending.

        Args:
            doc: Document to append to the index.

        Returns:
            self
        """
        doc_id = len(self.docs)
        self.docs.append(doc)
        self.total_docs += 1

        # Update inverted index for text fields
        for field in self.text_fields:
            self._update_inverted_index(doc_id, field, doc.get(field, ""))

        # Update keyword data
        for field in self.keyword_fields:
            self.keyword_data[field].append(doc.get(field))

        # Update numeric data
        for field in self.numeric_fields:
            self.numeric_data[field].append(doc.get(field))

        # Update date data
        for field in self.date_fields:
            value = doc.get(field)
            if isinstance(value, (date, datetime)):
                self.date_data[field].append(pd.Timestamp(value))
            else:
                self.date_data[field].append(value)

        # Recompute IDF values
        self._finalize_index()

        # Update the filter
        self._filter.refresh(
            keyword_data=self.keyword_data,
            numeric_data=self.numeric_data,
            date_data=self.date_data,
            num_docs=len(self.docs),
        )

        return self

    def search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        boost_dict: Optional[Dict] = None,
        num_results: int = 10,
        output_ids: bool = False,
    ) -> List[Dict]:
        """
        Search the index with the given query.

        Args:
            query: The search query string.
            filter_dict: Dictionary of keyword, numeric, and date fields to filter by.
            boost_dict: Dictionary of boost scores for text fields.
            num_results: The number of top results to return.
            output_ids: If True, adds an '_id' field to each document.

        Returns:
            List of documents matching the search criteria, ranked by relevance.
        """
        if filter_dict is None:
            filter_dict = {}
        if boost_dict is None:
            boost_dict = {}

        if not self.docs:
            return []

        query_tokens = self._process_text(query)
        if not query_tokens:
            return []

        scores = np.zeros(len(self.docs))

        # Calculate scores for each text field
        for field in self.text_fields:
            # Create query vector
            query_vector, field_tokens = self._create_query_vector(field, query_tokens)
            if query_vector is None:
                continue

            # Get matching documents (using sets, fast union)
            matching_docs = self._get_matching_documents(field, field_tokens)

            # Calculate field scores
            field_scores = self._calculate_field_scores(
                field, query_vector, field_tokens, matching_docs
            )

            # Apply boost
            boost = boost_dict.get(field, 1)
            scores += field_scores * boost

        # Apply filters using the Filter object
        filter_mask = self._filter.apply(filter_dict)
        scores = scores * filter_mask

        # Get top results
        non_zero_mask = scores > 0
        if not np.any(non_zero_mask):
            return []

        # Get indices of non-zero scores and sort
        non_zero_indices = np.where(non_zero_mask)[0]
        sorted_indices = non_zero_indices[np.argsort(-scores[non_zero_indices])]

        # Take top num_results
        top_indices = sorted_indices[:num_results]

        if output_ids:
            return [{**self.docs[i], "_id": int(i)} for i in top_indices]
        return [self.docs[i] for i in top_indices]
