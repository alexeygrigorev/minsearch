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
    A highlighter that processes search results, highlighting specified fields
    and passing through other fields.

    Args:
        highlight_fields: List of field names to extract highlights from.
        skip_fields: List of field names to exclude from output.
        max_matches: Maximum number of matches to return per field (default 5).
        snippet_size: Maximum characters per match snippet (default 200).
        highlight_format: Format for highlights. Can be:
            - str: Delimiter to wrap both sides ("**" -> **text**)
            - tuple: (open, close) delimiters (("[", "]") -> [text])
            - callable: lambda text: f"__{text}__" -> __text__

    Example:
        >>> from minsearch import Index, Highlighter
        >>> index = Index(text_fields=['question', 'text'])
        >>> index.fit(docs)
        >>> highlighter = Highlighter(
        ...     highlight_fields=['question', 'text'],
        ...     skip_fields=['course']
        ... )
        >>> results = index.search('python machine learning')
        >>> highlighted = highlighter.highlight('python machine learning', results)
        >>> print(highlighted[0])
        {
            'question': {
                'matches': ['How do I use **Python** for **machine learning**?'],
                'total_matches': 2
            },
            'text': {
                'matches': ['**Python** is a programming language...'],
                'total_matches': 3
            },
            'section': 'Programming'  # pass-through
        }
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
        highlight_fields: List[str],
        skip_fields: Optional[List[str]] = None,
        max_matches: int = 5,
        snippet_size: int = 200,
        highlight_format: HighlightFormat = "**",
        stop_words: Optional[Set[str]] = None,
    ):
        """
        Initialize the Highlighter.

        Args:
            highlight_fields: List of field names to extract highlights from.
            skip_fields: List of field names to exclude from output.
            max_matches: Maximum matches to return per field.
            snippet_size: Maximum characters per match snippet.
            highlight_format: Format for highlights (str, tuple, or callable).
            stop_words: Set of stop words to ignore. If None, uses defaults.
        """
        self.highlight_fields = highlight_fields
        self.skip_fields = set(skip_fields or [])
        self.max_matches = max_matches
        self.snippet_size = snippet_size
        self.highlight_format = highlight_format
        self.stop_words = self.DEFAULT_STOP_WORDS if stop_words is None else stop_words

    def _apply_highlight(self, text: str) -> str:
        """Apply highlight formatting to a piece of text."""
        fmt = self.highlight_format

        if callable(fmt):
            return fmt(text)

        if isinstance(fmt, str):
            return f"{fmt}{text}{fmt}"

        if isinstance(fmt, tuple) and len(fmt) == 2:
            open_delim, close_delim = fmt
            return f"{open_delim}{text}{close_delim}"

        return f"**{text}**"

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words, removing stop words."""
        pattern = re.compile(r"[\s\W\d]+")
        tokens = [token.lower() for token in pattern.split(text) if token]
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens

    def _extract_query_terms(self, query: str) -> List[str]:
        """
        Extract meaningful query terms (removing stop words).

        Preserves multi-word phrases and returns terms in order of appearance.
        """
        # First, try to extract phrases (words in quotes)
        phrases = []
        phrase_pattern = re.compile(r'"([^"]+)"')
        for match in phrase_pattern.finditer(query):
            phrases.append(match.group(1))

        # Tokenize the query (without quotes)
        query_no_quotes = phrase_pattern.sub('', query)
        tokens = self._tokenize(query_no_quotes)

        # Combine phrases and tokens, preserving order
        result = []
        query_lower = query.lower()

        # Add tokens that appear in the query
        for token in tokens:
            if token in query_lower:
                result.append(token)

        # Add phrases
        for phrase in phrases:
            phrase_lower = phrase.lower()
            # Only add if the phrase actually appears in the query
            if phrase_lower in query_lower:
                result.append(phrase_lower)

        return result

    def _find_match_positions(self, text: str, query_terms: List[str]) -> List[tuple[int, int, str]]:
        """Find all positions where query terms appear in text."""
        if not text:
            return []

        text_lower = text.lower()
        matches = []

        for term in query_terms:
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                # Check word boundary
                before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
                after_ok = (
                    pos + len(term) >= len(text_lower)
                    or not text_lower[pos + len(term)].isalnum()
                )
                if before_ok and after_ok:
                    matches.append((pos, pos + len(term), term))
                start = pos + 1

        matches.sort(key=lambda x: x[0])
        return matches

    def _create_snippet(self, text: str, matches: List[tuple[int, int, str]]) -> str:
        """Create a highlighted snippet from a window of text with matches."""
        if not matches:
            return ""

        # Find the bounds of the snippet centered on first match
        first_start, first_end, _ = matches[0]
        window_start = max(0, first_start - self.snippet_size // 2)
        window_end = min(len(text), first_end + self.snippet_size // 2)

        # Collect all matches within this window
        window_matches = [
            m for m in matches
            if window_start <= m[0] < window_end
        ]

        # Extract the window text
        result = text[window_start:window_end]

        # Apply highlights (reverse order to preserve positions)
        for start, end, _ in sorted(window_matches, key=lambda x: -x[0]):
            rel_start = start - window_start
            rel_end = end - window_start
            actual_text = text[start:end]
            highlighted = self._apply_highlight(actual_text)
            result = result[:rel_start] + highlighted + result[rel_end:]

        # Add ellipsis
        prefix = "..." if window_start > 0 else ""
        suffix = "..." if window_end < len(text) else ""

        return prefix + result + suffix

    def _highlight_field(self, text: str, query_terms: List[str]) -> Dict:
        """
        Extract highlighted snippets from a field.

        Returns:
            Dict with 'matches' (list of snippets) and 'total_matches' (count).
        """
        if not text:
            return {"matches": [], "total_matches": 0}

        matches = self._find_match_positions(text, query_terms)

        if not matches:
            return {"matches": [], "total_matches": 0}

        # Create snippets
        snippets = []
        used_matches = []

        for match in matches:
            if len(snippets) >= self.max_matches:
                break
            # Check if this match overlaps with already used matches
            start, end, _ = match
            overlaps = any(
                start < used_end and end > used_start
                for used_start, used_end, _ in used_matches
            )
            if not overlaps:
                snippet = self._create_snippet(text, [match] + [
                    m for m in matches
                    if abs(m[0] - start) < self.snippet_size and m is not match
                ])
                if snippet:
                    snippets.append(snippet)
                    used_matches.append(match)

        return {
            "matches": snippets,
            "total_matches": len(matches)
        }

    def highlight(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Process search results, highlighting specified fields.

        Args:
            query: The search query (can be natural language).
            results: List of documents from search.

        Returns:
            List of dictionaries with highlighted fields and pass-through fields.
        """
        query_terms = self._extract_query_terms(query)

        highlighted_results = []

        for doc in results:
            result = {}

            for field, value in doc.items():
                if field in self.skip_fields:
                    continue

                if field in self.highlight_fields:
                    # Highlight this field
                    text = str(value) if value is not None else ""
                    if query_terms:
                        result[field] = self._highlight_field(text, query_terms)
                    else:
                        # No query terms - return empty structure
                        result[field] = {"matches": [], "total_matches": 0}
                else:
                    # Pass-through
                    result[field] = value

            highlighted_results.append(result)

        return highlighted_results
