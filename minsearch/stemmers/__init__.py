"""
Stemmers for text normalization.

This package provides various stemmer implementations for reducing words
to their root form, which can improve search matching by finding related
words (e.g., "running", "runs", "ran" all stem to "run").

Available stemmers:
    - porter: Porter stemmer (most common, conservative)
    - snowball: Snowball stemmer (more aggressive than Porter)
    - lancaster: Lancaster stemmer (very aggressive)

Example:
    >>> from minsearch.stemmers import porter_stemmer, get_stemmer
    >>> porter_stemmer("running")
    'run'
    >>> stemmer = get_stemmer("porter")
    >>> stemmer("jumped")
    'jump'
"""

from .porter import porter_stemmer
from .snowball import snowball_stemmer
from .lancaster import lancaster_stemmer
from .base import get_stemmer, STEMMERS

__all__ = [
    "porter_stemmer",
    "snowball_stemmer",
    "lancaster_stemmer",
    "get_stemmer",
    "STEMMERS",
]
