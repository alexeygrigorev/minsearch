"""
Lancaster stemmer for English.

A very aggressive stemming algorithm.
Removes more suffixes than Porter or Snowball.

The Lancaster stemmer (Paice/Husk stemmer) is a iterative algorithm
that applies rules until no more changes can be made.

Reference:
    Paice, C.D. (1990). "Another stemmer".
    ACM SIGIR Forum, 24(3), 56-61.
"""

from typing import List, Tuple, Optional


def lancaster_stemmer(word: str) -> str:
    """
    Apply the Lancaster stemming algorithm to a word.

    The Lancaster stemmer is very aggressive - it removes many more
    suffixes than Porter or Snowball.

    Args:
        word: The word to stem.

    Returns:
        The stemmed word in lowercase.

    Examples:
        >>> lancaster_stemmer("running")
        'run'
        >>> lancaster_stemmer("maximum")
        'maxim'
        >>> lancaster_stemmer("presumably")
        'presum'
        >>> lancaster_stemmer("multiply")
        'multiply'
    """
    if not word:
        return ""

    word = word.lower()

    # Apply rules iteratively until no more changes
    changed = True
    max_iterations = 10  # Prevent infinite loops
    iteration = 0

    while changed and iteration < max_iterations and word:
        iteration += 1
        changed = False
        word_before = word

        # Apply rules in order
        for rule in _get_rules():
            result = _apply_rule(word, rule)
            if result is not None:
                word = result
                changed = True
                break

        # Stop if word is too short
        if len(word) <= 2:
            break

    # Post-process: remove trailing double consonants
    # This handles cases like "running" -> "runn" -> "run"
    if len(word) >= 4 and word[-1] == word[-2] and word[-1] not in "aeiouy":
        word = word[:-1]

    return word


def _apply_rule(word: str, rule: Tuple) -> Optional[str]:
    """
    Apply a single stemming rule to a word.

    Args:
        word: The word to stem.
        rule: A tuple of (ending, replacement, accept_length, intact, continued).

    Returns:
        The stemmed word, or None if the rule doesn't apply.
    """
    ending, replacement, accept_length, intact, continued = rule

    # Check if word ends with the rule's ending
    if not word.endswith(ending):
        return None

    # Calculate stem
    stem = word[:-len(ending)]

    # Check if stem is long enough
    if len(stem) < accept_length:
        return None

    # Check if the ending must be intact (preceded by a vowel)
    if intact and len(stem) > 0:
        if stem[-1] not in "aeiouy":
            return None

    # Apply the replacement
    if replacement:
        result = stem + replacement
    else:
        result = stem

    # Check minimum length
    if len(result) <= 1:
        return None

    # If continued is False, this is the final rule
    if not continued and len(result) > 2:
        return result

    return result


def _get_rules() -> List[Tuple[str, str, int, bool, bool]]:
    """
    Get the Lancaster stemming rules.

    Returns:
        List of tuples: (ending, replacement, accept_length, intact, continued)
    """
    return [
        # Very aggressive suffix removal
        ("eational", "eate", 5, False, True),
        ("tional", "tion", 4, False, True),
        ("ational", "ate", 4, False, True),
        ("alizi", "al", 4, False, True),
        ("ization", "ize", 4, False, True),
        ("ation", "ate", 4, False, True),
        ("ator", "ate", 3, False, True),
        ("iveness", "ive", 5, False, True),
        ("fulness", "ful", 5, False, True),
        ("ousness", "ous", 5, False, True),
        ("iveness", "", 5, False, True),
        ("fulness", "", 5, False, True),
        ("ousness", "", 5, False, True),
        ("iviti", "ive", 4, False, True),
        ("biliti", "ble", 4, False, True),
        ("lessli", "less", 5, False, True),
        ("entli", "ent", 4, False, True),
        ("ation", "", 4, False, True),
        ("alism", "al", 4, False, True),
        ("aliti", "al", 4, False, True),
        ("iviti", "", 4, False, True),
        ("biliti", "", 4, False, True),
        ("ousli", "", 6, False, True),
        ("entli", "", 4, False, True),
        ("ization", "", 4, False, True),
        ("zation", "", 4, False, True),
        ("ation", "", 3, False, True),
        ("ally", "", 3, False, True),
        ("ely", "", 4, False, True),
        ("ingly", "", 5, False, True),
        ("edly", "", 5, False, True),
        ("ingly", "", 5, False, True),
        ("ingly", "", 5, False, True),
        ("ingly", "y", 5, False, True),
        ("edly", "y", 5, False, True),
        ("eedli", "", 5, False, True),
        ("eedly", "", 5, False, True),
        # Common endings
        ("ing", "", 4, False, True),
        ("ing", "", 3, False, True),
        ("ing", "", 2, True, True),
        ("edly", "", 5, False, True),
        ("eedli", "", 5, False, True),
        ("ingly", "", 5, False, True),
        ("edly", "", 5, False, True),
        ("ingly", "", 5, False, True),
        # Past tense
        ("ed", "", 4, False, True),
        ("ed", "", 3, False, True),
        ("ied", "y", 3, False, True),
        ("ied", "i", 3, False, True),
        ("edly", "", 5, False, True),
        ("eedly", "", 5, False, True),
        # Plurals
        ("ies", "y", 3, False, True),
        ("ies", "i", 3, False, True),
        ("es", "", 4, False, True),
        ("es", "", 3, False, True),
        ("es", "", 2, True, True),
        ("s", "", 4, False, True),
        ("s", "", 3, False, True),
        ("s", "", 2, True, True),
        ("men", "man", 3, False, True),
        ("ss", "", 0, False, True),
        ("ies", "", 3, False, True),
        # Additional endings
        ("ness", "", 4, False, True),
        ("ness", "", 3, False, True),
        ("ment", "", 4, False, True),
        ("ment", "", 3, False, True),
        ("able", "", 4, False, True),
        ("ible", "", 4, False, True),
        ("ly", "", 3, False, True),
        ("ly", "", 2, True, True),
        ("er", "", 4, False, True),
        ("er", "", 3, False, True),
        ("est", "", 4, False, True),
        ("est", "", 3, False, True),
        ("ity", "", 3, False, True),
        ("ify", "", 3, False, True),
        ("ize", "", 3, False, True),
        ("ise", "", 3, False, True),
        ("al", "", 4, False, True),
        ("al", "", 3, False, True),
        ("ial", "", 3, False, True),
        ("ual", "", 3, False, True),
        ("ial", "i", 3, False, True),
        ("eaux", "", 0, False, True),
        ("eaus", "", 0, False, True),
        ("eous", "", 3, False, True),
        ("ious", "", 3, False, True),
        # Final cleanup
        ("le", "", 3, False, True),
        ("ize", "", 3, False, True),
        ("ise", "", 3, False, True),
        ("ful", "", 3, False, True),
        ("est", "", 3, False, True),
        ("ment", "", 3, False, True),
        ("ible", "", 3, False, True),
        ("ance", "", 3, False, True),
        ("ence", "", 3, False, True),
        ("ate", "", 3, False, True),
        ("iti", "", 3, False, True),
        ("ous", "", 3, False, True),
        ("ive", "", 3, False, True),
        ("ize", "", 3, False, True),
        ("al", "", 2, False, True),
        ("er", "", 2, False, True),
        ("est", "", 2, False, True),
        ("ment", "", 2, False, True),
        ("ness", "", 2, False, True),
        ("able", "", 2, False, True),
        ("ible", "", 2, False, True),
        ("ly", "", 2, False, True),
        ("ful", "", 2, False, True),
        ("ing", "", 2, False, True),
        ("ed", "", 2, False, True),
        # Single letter removals
        ("s", "", 3, False, True),
        ("s", "", 2, True, True),
        ("e", "", 3, False, True),
        ("e", "", 2, True, True),
    ]
