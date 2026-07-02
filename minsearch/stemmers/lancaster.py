"""
Lancaster stemmer (compatibility re-export).

The implementation now lives in the shared ``stemlite`` library. This module
re-exports it so that ``from minsearch.stemmers.lancaster import lancaster_stemmer``
keeps working.
"""

from stemlite.lancaster import lancaster_stemmer

__all__ = ["lancaster_stemmer"]
