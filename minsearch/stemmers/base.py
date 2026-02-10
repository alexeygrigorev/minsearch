"""
Base stemmer utilities.
"""

from typing import Callable, Optional, Dict

from .porter import porter_stemmer
from .snowball import snowball_stemmer
from .lancaster import lancaster_stemmer


# Registry of available stemmers
STEMMERS: Dict[str, Callable[[str], str]] = {
    "porter": porter_stemmer,
    "snowball": snowball_stemmer,
    "lancaster": lancaster_stemmer,
    "none": lambda w: w.lower() if w else "",
}


def get_stemmer(name: Optional[str] = None) -> Callable[[str], str]:
    """
    Get a stemmer by name.

    Args:
        name: Name of the stemmer ('porter', 'snowball', 'lancaster', or None).
              If None, returns a no-op stemmer (lowercase only).

    Returns:
        A stemmer function that takes a word and returns its stemmed form.

    Example:
        >>> get_stemmer("porter")("running")
        'run'
        >>> get_stemmer(None)("Running")
        'running'
    """
    if name is None:
        return STEMMERS["none"]
    name_lower = name.lower()
    return STEMMERS.get(name_lower, STEMMERS["none"])
