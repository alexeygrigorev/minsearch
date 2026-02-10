import re
from typing import Dict, List, Optional, Set, Callable, Union, Tuple
from collections import Counter


HighlightFormat = Union[
    str,  # Delimiter: "**" -> **text**
    Tuple[str, str],  # Open/close: ("[", "]") -> [text]
    Callable[[str], str],  # Custom function: lambda t: f"__{t}__" -> __text__
]


class Highlighter:
    """
    A highlighter that extracts relevant snippets from search results.

    Instead of returning full documents, it returns only the portions
    of text that match the query terms, making it easier to see why
    a document was returned.

    Args:
        text_fields: List of field names to extract highlights from.
        snippet_size: Maximum number of characters per snippet.
        max_snippets: Maximum number of snippets per document.
        highlight_format: Format for highlights. Can be:
            - str: Delimiter to wrap both sides ("**" -> **text**)
            - tuple: (open, close) delimiters (("[", "]") -> [text])
            - callable: lambda text: f"__{text}__" -> __text__
        stop_words: Set of stop words to ignore.

    Example:
        >>> from minsearch import Index, Highlighter
        >>> index = Index(text_fields=['text', 'title'])
        >>> index.fit(docs)
        >>> highlighter = Highlighter(text_fields=['text', 'title'])
        >>> results = index.search('search query')
        >>> highlighted = highlighter.highlight('search query', results)
        >>> print(highlighted[0]['highlights']['text'])
        ['...this is a **search** **query** example...']

        Custom format:
        >>> highlighter = Highlighter(text_fields=['text'], highlight_format='__')
        >>> # Returns: '...this is a __search__ __query__ example...'
    """

    DEFAULT_STOP_WORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can't",
        "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from",
        "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
        "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
        "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
        "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
        "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
        "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves",
    }

    def __init__(
        self,
        text_fields: List[str],
        snippet_size: int = 200,
        max_snippets: int = 3,
        highlight_format: HighlightFormat = "**",
        stop_words: Optional[Set[str]] = None,
    ):
        """
        Initialize the Highlighter.

        Args:
            text_fields: List of field names to extract highlights from.
            snippet_size: Maximum characters per snippet (window around match).
            max_snippets: Maximum snippets per document per field.
            highlight_format: Format for highlights (str, tuple, or callable).
            stop_words: Set of stop words to ignore. If None, uses defaults.
        """
        self.text_fields = text_fields
        self.snippet_size = snippet_size
        self.max_snippets = max_snippets
        self.highlight_format = highlight_format
        self.stop_words = self.DEFAULT_STOP_WORDS if stop_words is None else stop_words

    def _apply_highlight(self, text: str) -> str:
        """
        Apply highlight formatting to a piece of text.

        Args:
            text: The text to highlight.

        Returns:
            Formatted text with highlights applied.
        """
        fmt = self.highlight_format

        if callable(fmt):
            return fmt(text)

        if isinstance(fmt, str):
            # Single delimiter wraps both sides
            return f"{fmt}{text}{fmt}"

        if isinstance(fmt, tuple) and len(fmt) == 2:
            # Tuple with open and close delimiters
            open_delim, close_delim = fmt
            return f"{open_delim}{text}{close_delim}"

        # Fallback to default
        return f"**{text}**"

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words, removing stop words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        pattern = re.compile(r"[\s\W\d]+")
        tokens = [token.lower() for token in pattern.split(text) if token]
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def _extract_query_terms(self, query: str) -> Set[str]:
        """
        Extract meaningful query terms (removing stop words).

        Args:
            query: Search query string.

        Returns:
            Set of query terms to highlight.
        """
        return set(self._tokenize(query))

    def _find_match_positions(
        self, text: str, query_terms: Set[str]
    ) -> List[tuple[int, int, str]]:
        """
        Find all positions where query terms appear in text.

        Args:
            text: The text to search in.
            query_terms: Set of query terms to find.

        Returns:
            List of (start, end, matched_term) tuples, sorted by position.
        """
        if not text:
            return []

        text_lower = text.lower()
        matches = []

        for term in query_terms:
            # Find all occurrences of this term
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                # Check if it's a whole word match (word boundary before and after)
                before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
                after_ok = (
                    pos + len(term) >= len(text_lower)
                    or not text_lower[pos + len(term)].isalnum()
                )
                if before_ok and after_ok:
                    matches.append((pos, pos + len(term), term))
                start = pos + 1

        # Sort by position
        matches.sort(key=lambda x: x[0])
        return matches

    def _merge_overlapping_snippets(
        self, matches: List[tuple[int, int, str]], text: str
    ) -> List[str]:
        """
        Merge overlapping or nearby matches into snippets.

        Args:
            matches: List of (start, end, term) tuples.
            text: Original text.

        Returns:
            List of highlighted snippet strings.
        """
        if not matches:
            return []

        snippets = []
        i = 0

        while i < len(matches) and len(snippets) < self.max_snippets:
            start, end, _ = matches[i]

            # Expand window to snippet_size
            window_start = max(0, start - self.snippet_size // 2)
            window_end = min(len(text), end + self.snippet_size // 2)

            # Collect all matches within this window
            window_matches = []
            j = i
            while j < len(matches) and matches[j][0] < window_end:
                window_matches.append(matches[j])
                j += 1

            # Create snippet with highlights
            snippet = self._create_highlighted_snippet(
                text, window_matches, window_start, window_end
            )
            snippets.append(snippet)

            # Move to next window (skip matches we already included)
            i = j

        return snippets

    def _create_highlighted_snippet(
        self,
        text: str,
        matches: List[tuple[int, int, str]],
        window_start: int,
        window_end: int,
    ) -> str:
        """
        Create a highlighted snippet from text with matches.

        Args:
            text: Original text.
            matches: List of (start, end, term) tuples within the window.
            window_start: Start position of window.
            window_end: End position of window.

        Returns:
            Highlighted snippet string.
        """
        result = text[window_start:window_end]

        # Sort matches by position (reverse order to avoid offset issues)
        sorted_matches = sorted(matches, key=lambda x: -x[0])

        # Apply highlights (from end to start to preserve positions)
        for start, end, term in sorted_matches:
            # Adjust positions relative to window
            rel_start = start - window_start
            rel_end = end - window_start

            # Extract the actual matched text (preserve original case)
            actual_text = text[start:end]

            # Apply highlight format
            highlighted = self._apply_highlight(actual_text)
            result = result[:rel_start] + highlighted + result[rel_end:]

        # Add ellipsis if needed
        prefix = "..." if window_start > 0 else ""
        suffix = "..." if window_end < len(text) else ""

        return prefix + result + suffix

    def _highlight_field(
        self, text: str, query_terms: Set[str]
    ) -> List[str]:
        """
        Extract highlighted snippets from a single field.

        Args:
            text: Text to highlight.
            query_terms: Query terms to look for.

        Returns:
            List of highlighted snippets.
        """
        if not text:
            return []

        matches = self._find_match_positions(text, query_terms)

        if not matches:
            return []

        return self._merge_overlapping_snippets(matches, text)

    def highlight(
        self,
        query: str,
        results: List[Dict],
        fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Extract highlights from search results.

        Args:
            query: The search query that produced these results.
            results: List of documents from search.
            fields: Optional list of fields to highlight. If None, uses text_fields from init.

        Returns:
            List of dictionaries with 'highlights' key containing highlighted snippets.
            Each result has: {'highlights': {field: [snippets], ...}, 'document': {...}}
        """
        if fields is None:
            fields = self.text_fields

        query_terms = self._extract_query_terms(query)

        if not query_terms:
            # No meaningful terms, return empty highlights
            return [{"highlights": {}, "document": doc} for doc in results]

        highlighted_results = []

        for doc in results:
            highlights = {}

            for field in fields:
                if field in doc and doc[field]:
                    text = str(doc[field])
                    snippets = self._highlight_field(text, query_terms)
                    if snippets:
                        highlights[field] = snippets

            highlighted_results.append({"highlights": highlights, "document": doc})

        return highlighted_results
