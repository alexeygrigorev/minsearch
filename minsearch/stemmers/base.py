"""
Base stemmer utilities (compatibility re-export).

The implementation now lives in the shared ``stemlite`` library. This module
re-exports it so that ``from minsearch.stemmers.base import get_stemmer, STEMMERS``
keeps working.
"""

from stemlite.base import _none_stemmer, get_stemmer, STEMMERS

__all__ = ["get_stemmer", "STEMMERS", "_none_stemmer"]
