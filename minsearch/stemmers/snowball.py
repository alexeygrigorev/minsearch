"""
Snowball stemmer (compatibility re-export).

The implementation now lives in the shared ``stemlite`` library. This module
re-exports it so that ``from minsearch.stemmers.snowball import snowball_stemmer``
keeps working.
"""

from stemlite.snowball import snowball_stemmer

__all__ = ["snowball_stemmer"]
