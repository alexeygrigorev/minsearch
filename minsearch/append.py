import re
import math

from collections import defaultdict

import numpy as np


class Tokenizer:
    """
    A custom tokenizer that splits text into tokens and removes stop words.
    Mimics sklearn's default tokenizer behavior.
    """

    # Common English stop words (similar to sklearn's default)
    DEFAULT_STOP_WORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }

    def __init__(self, pattern=r"[\s\W\d]+", stop_words=None):
        """
        Initialize the tokenizer with a regex pattern and stop words.

        Args:
            pattern (str): Regex pattern to split text on. Default pattern splits on
                          whitespace, non-word characters, and digits.
            stop_words (set or None): Set of stop words to remove. If None, uses default
                                    English stop words. If empty set, no stop words are removed.
        """
        self.pattern = pattern
        self.stop_words = self.DEFAULT_STOP_WORDS if stop_words is None else stop_words

    def tokenize(self, text):
        """
        Tokenize the input text and remove stop words.

        Args:
            text (str): Text to tokenize

        Returns:
            list: List of tokens with stop words removed
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Split on the pattern and filter out empty strings
        tokens = [token for token in re.split(self.pattern, text) if token]

        # Remove stop words if any are specified
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        return tokens


class AppendableIndex:
    """
    An appendable search index using inverted index for text fields and exact matching for keyword fields.
    Maintains the same interface as the original Index class but allows for appending documents.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        keyword_data (dict): Dictionary containing keyword field data as lists.
        docs (list): List of documents indexed.
        inverted_index (dict): Dictionary of inverted indices for each text field.
        doc_frequencies (dict): Dictionary of document frequencies for each text field.
        total_docs (int): Total number of documents in the index.
    """

    def __init__(self, text_fields, keyword_fields=None, stop_words=None):
        """
        Initializes the AppendableIndex with specified text and keyword fields.

        Args:
            text_fields (list): List of text field names to index.
            keyword_fields (list, optional): List of keyword field names to index. Defaults to empty list.
            stop_words (set or None): Set of stop words to remove. If None, uses default
                                    English stop words. If empty set, no stop words are removed.
        """
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields if keyword_fields is not None else []

        # Initialize data structures
        self.docs = []
        self.total_docs = 0
        self.inverted_index = {field: defaultdict(list) for field in text_fields}
        self.doc_frequencies = {field: defaultdict(int) for field in text_fields}
        self.keyword_data = {field: [] for field in self.keyword_fields}

        # Store vocabulary for each field
        self.vocabularies = {field: set() for field in text_fields}

        # Initialize tokenizer with stop words
        self.tokenizer = Tokenizer(stop_words=stop_words)

    def _process_text(self, text):
        """Process text into tokens using our custom tokenizer."""
        return self.tokenizer.tokenize(text)

    def _update_inverted_index(self, doc_id, field, text):
        """Update the inverted index for a given field and document."""
        tokens = self._process_text(text)
        if not tokens:  # Skip empty documents
            return

        for token in tokens:
            self.inverted_index[field][token].append(doc_id)
            self.doc_frequencies[field][token] = len(
                set(self.inverted_index[field][token])
            )
            self.vocabularies[field].add(token)

    def _calculate_tfidf(self, field, token, doc_id):
        """Calculate TF-IDF score for a token in a document."""
        # Term frequency (TF)
        doc_tokens = self._process_text(self.docs[doc_id].get(field, ""))
        if not doc_tokens:  # Handle empty documents
            return 0

        # Use sublinear TF scaling like scikit-learn
        tf = 1 + math.log(doc_tokens.count(token)) if doc_tokens.count(token) > 0 else 0

        # Inverse document frequency (IDF)
        df = self.doc_frequencies[field][token]
        idf = math.log((self.total_docs + 1) / (df + 1)) + 1

        return tf * idf

    def _get_matching_documents(self, field, tokens):
        """Get set of document IDs that match any of the given tokens in the field."""
        matching_docs = set()
        for token in tokens:
            if token in self.inverted_index[field]:
                matching_docs.update(self.inverted_index[field][token])
        return matching_docs

    def _create_query_vector(self, field, query_tokens):
        """Create and normalize TF-IDF vector for the query in the given field."""
        # Get unique query tokens that exist in this field
        field_tokens = [t for t in query_tokens if t in self.inverted_index[field]]
        if not field_tokens:
            return None, None

        # Calculate query vector
        query_vector = np.zeros(len(field_tokens))
        for i, token in enumerate(field_tokens):
            # Calculate TF-IDF for query token
            tf = (
                1 + math.log(query_tokens.count(token))
                if query_tokens.count(token) > 0
                else 0
            )
            df = self.doc_frequencies[field][token]
            idf = math.log((self.total_docs + 1) / (df + 1)) + 1
            query_vector[i] = tf * idf

        # L2 normalize the query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        return query_vector, field_tokens

    def _create_document_vectors(self, field, field_tokens, matching_docs):
        """Create and normalize TF-IDF vectors for matching documents in the given field."""
        doc_vectors = {}
        for doc_id in matching_docs:
            doc_vector = np.zeros(len(field_tokens))
            doc_tokens = self._process_text(self.docs[doc_id].get(field, ""))
            if doc_tokens:
                # Calculate TF-IDF for each matching token
                for i, token in enumerate(field_tokens):
                    if token in doc_tokens:
                        tfidf = self._calculate_tfidf(field, token, doc_id)
                        doc_vector[i] = tfidf

                # Calculate the FULL document norm using ALL tokens (not just query-matching)
                # This matches sklearn's behavior where normalization considers all terms
                full_doc_norm_squared = 0.0
                # Use set to avoid double-counting tokens (TF-IDF already accounts for term frequency)
                unique_doc_tokens = set(doc_tokens)
                for token in unique_doc_tokens:
                    tfidf = self._calculate_tfidf(field, token, doc_id)
                    full_doc_norm_squared += tfidf * tfidf
                full_doc_norm = math.sqrt(full_doc_norm_squared)

                # L2 normalize using the full document norm
                if full_doc_norm > 0:
                    doc_vector = doc_vector / full_doc_norm
                doc_vectors[doc_id] = doc_vector
        return doc_vectors

    def _calculate_field_scores(self, query_vector, doc_vectors):
        """Calculate cosine similarity scores between query and document vectors."""
        field_scores = np.zeros(len(self.docs))
        for doc_id, doc_vector in doc_vectors.items():
            field_scores[doc_id] = np.dot(query_vector, doc_vector)
        return field_scores

    def _apply_keyword_filters(self, scores, filter_dict):
        """Apply keyword filters to the scores."""
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                if value is None:
                    mask = np.array([val is None for val in self.keyword_data[field]])
                else:
                    mask = np.array([val == value for val in self.keyword_data[field]])
                scores = scores * mask
        return scores

    def _get_top_results(self, scores, num_results):
        """Get top scoring documents based on the scores."""
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
        return sorted_indices[:num_results]

    def fit(self, docs):
        """
        Fits the index with the provided documents.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
        """
        self.docs = docs
        self.total_docs = len(docs)

        # Process each document
        for doc_id, doc in enumerate(docs):
            # Update inverted index for text fields
            for field in self.text_fields:
                self._update_inverted_index(doc_id, field, doc.get(field, ""))

            # Collect keyword data
            for field in self.keyword_fields:
                self.keyword_data[field].append(doc.get(field))

        # Only check vocabulary if we have documents
        if self.docs:
            has_vocabulary = any(len(vocab) > 0 for vocab in self.vocabularies.values())
            if not has_vocabulary:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )

        return self

    def append(self, doc):
        """
        Appends a single document to the index.

        Args:
            doc (dict): Document to append to the index.
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

        return self

    def search(
        self, query, filter_dict=None, boost_dict=None, num_results=10, output_ids=False
    ):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields.
            num_results (int): The number of top results to return.
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

        query_tokens = self._process_text(query)
        if not query_tokens:  # Handle empty queries
            return []

        scores = np.zeros(len(self.docs))

        # Calculate scores for each text field
        for field in self.text_fields:
            # Create query vector
            query_vector, field_tokens = self._create_query_vector(field, query_tokens)
            if query_vector is None:
                continue

            # Get matching documents
            matching_docs = self._get_matching_documents(field, field_tokens)

            # Create document vectors
            doc_vectors = self._create_document_vectors(
                field, field_tokens, matching_docs
            )

            # Calculate field scores
            field_scores = self._calculate_field_scores(query_vector, doc_vectors)

            # Apply boost
            boost = boost_dict.get(field, 1)
            scores += field_scores * boost

        # Apply keyword filters
        scores = self._apply_keyword_filters(scores, filter_dict)

        # Get top results
        top_indices = self._get_top_results(scores, num_results)

        if output_ids:
            return [{**self.docs[i], "_id": int(i)} for i in top_indices]
        return [self.docs[i] for i in top_indices]
