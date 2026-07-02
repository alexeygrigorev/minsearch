"""
Porter stemmer (compatibility re-export).

The implementation now lives in the shared ``stemlite`` library. This module
re-exports it so that ``from minsearch.stemmers.porter import porter_stemmer``
keeps working.
"""

from stemlite.porter import porter_stemmer

__all__ = ["porter_stemmer"]
