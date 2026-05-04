"""
Tokenizer module for text processing.

A custom tokenizer that splits text into tokens and removes stop words.
Mimics sklearn's default tokenizer behavior.
"""

import re
from pathlib import Path
from typing import List, Set, Optional, Union, Callable, Literal

from .stemmers import get_stemmer


StopWordsOption = Union[Literal['english'], Set[str]]


def _load_stop_words() -> Set[str]:
    """Load stop words from stop_words.txt file."""
    module_dir = Path(__file__).parent
    stop_words_path = module_dir / "stop_words.txt"
    with open(stop_words_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


# Load English stop words from file
DEFAULT_ENGLISH_STOP_WORDS = _load_stop_words()


class Tokenizer:
    """
    A custom tokenizer that splits text into tokens, removes stop words,
    and optionally applies stemming.

    Mimics sklearn's default tokenizer behavior with optional stemming.
    """

    def __init__(
        self,
        pattern: str = r"[\s\W\d]+",
        stop_words: Optional[StopWordsOption] = None,
        stemmer: Optional[Union[str, Callable[[str], str]]] = None,
        min_token_length: int = 2,
    ):
        """
        Initialize the tokenizer with a regex pattern and stop words.

        Args:
            pattern: Regex pattern to split text on.
            stop_words: Stop words to remove. Can be:
                - None: No stop words removed (default)
                - 'english': Use default English stop words from stop_words.txt
                - Set[str]: Custom set of stop words
            stemmer: Stemmer to use. Can be:
                - None: No stemming (default)
                - str: Name of built-in stemmer ('porter', 'snowball', 'lancaster')
                - callable: Custom stemmer function (word -> stemmed_word)
            min_token_length: Minimum token length to keep. Defaults to 2 to
                match sklearn's default TfidfVectorizer token pattern.
        """
        self.pattern = re.compile(pattern)
        self.min_token_length = min_token_length
        if stop_words == 'english':
            self.stop_words = DEFAULT_ENGLISH_STOP_WORDS
        elif stop_words is None:
            self.stop_words = set()
        else:
            self.stop_words = stop_words

        # Set up stemmer
        if stemmer is None or isinstance(stemmer, str):
            self.stemmer = get_stemmer(stemmer)
        else:
            self.stemmer = stemmer

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text and remove stop words.

        Applies stemming if a stemmer is configured.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens with stop words removed and stemmed if configured
        """
        if not text:
            return []

        text = text.lower()
        tokens = []
        for token in self.pattern.split(text):
            if not token:
                continue
            if len(token) < self.min_token_length:
                continue
            if token in self.stop_words:
                continue
            if self.stemmer:
                token = self.stemmer(token)
            tokens.append(token)

        return tokens
