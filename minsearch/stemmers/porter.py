"""
Porter stemmer for English.

A simplified implementation of the Porter stemming algorithm.
This is the most commonly used stemmer for English text.

The Porter stemmer is a conservative stemmer that follows a series
of rules to remove suffixes from words.

Reference:
    Porter, M.F. (1980). "An algorithm for suffix stripping".
    Program, 14(3), 130-137.
"""

from typing import Set


# Vowels for the algorithm
_VOWELS: Set[str] = {'a', 'e', 'i', 'o', 'u', 'y'}


def porter_stemmer(word: str) -> str:
    """
    Apply the Porter stemming algorithm to a word.

    This is a simplified implementation that handles common English patterns.

    Args:
        word: The word to stem.

    Returns:
        The stemmed word in lowercase.

    Examples:
        >>> porter_stemmer("running")
        'run'
        >>> porter_stemmer("jumps")
        'jump'
        >>> porter_stemmer("happiness")
        'happi'
    """
    if not word:
        return ""

    word = word.lower()

    # Step 1a: Handle plurals and past tense
    word = _step_1a(word)

    # Step 1b: Handle -eed, -ed, -ing endings
    word = _step_1b(word)

    # Step 1c: Handle -iveness, -fulness, etc.
    word = _step_1c(word)

    # Step 2: Handle various suffixes
    word = _step_2(word)

    # Step 3: Handle -ic-, -full, -ness, etc.
    word = _step_3(word)

    # Step 4: Handle -ive, -ment, etc.
    word = _step_4(word)

    # Step 5a: Handle -e endings
    word = _step_5a(word)

    # Step 5b: Handle double consonants at the end
    word = _step_5b(word)

    return word


def _contains_vowel(word: str, vowels: Set[str] = _VOWELS) -> bool:
    """Check if a word contains a vowel."""
    return any(char in vowels for char in word)


def _measure(word: str, vowels: Set[str] = _VOWELS) -> int:
    """
    Calculate the measure of a word (number of consonant-vowel sequences).

    [C](VC)^m[V] where C is a consonant and V is a vowel.
    """
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if not is_vowel and prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    return count


def _ends_double_consonant(word: str, vowels: Set[str] = _VOWELS) -> bool:
    """Check if word ends with a double consonant."""
    if len(word) < 2:
        return False
    return word[-1] == word[-2] and word[-1] not in vowels


def _ends_cvc(word: str, vowels: Set[str] = _VOWELS) -> bool:
    """Check if word ends with consonant-vowel-consonant pattern."""
    if len(word) < 3:
        return False
    return (word[-3] not in vowels and
            word[-2] in vowels and
            word[-1] not in vowels)


def _step_1a(word: str) -> str:
    """Step 1a: Handle plurals and past tense."""
    # SSES -> SS
    if word.endswith("sses"):
        return word[:-2]

    # IES -> I
    if word.endswith("ies"):
        return word[:-2]

    # SS -> SS
    if word.endswith("ss"):
        return word

    # S -> (remove)
    if word.endswith("s"):
        return word[:-1]

    return word


def _step_1b(word: str) -> str:
    """Step 1b: Handle -eed, -ed, -ing endings."""
    vowels = _VOWELS

    # Handle -eed
    if word.endswith("eed"):
        stem = word[:-3]
        if _measure(stem) > 0:
            return stem + "ee"
        return word

    # Handle -ed or -ing (only if stem contains vowel)
    has_ed_ing = False
    if word.endswith("ed"):
        stem = word[:-2]
        if _contains_vowel(stem, vowels):
            word = stem
            has_ed_ing = True
    elif word.endswith("ing"):
        stem = word[:-3]
        if _contains_vowel(stem, vowels):
            word = stem
            has_ed_ing = True

    if has_ed_ing:
        # If word ends in 'at', 'bl', or 'iz', add 'e'
        if word.endswith(("at", "bl", "iz")):
            return word + "e"

        # If word ends in double consonant and m>1, remove last consonant
        if _ends_double_consonant(word, vowels) and _measure(word, vowels) > 1:
            return word[:-1]

        # Special case: if word ends in double consonant and m==1, still remove one
        # This handles common cases like "running" -> "run" instead of "runn"
        if _ends_double_consonant(word, vowels) and _measure(word, vowels) == 1:
            if len(word) >= 3:  # Ensure we don't make it too short
                return word[:-1]

        # If m==1 and ends in cvc, add 'e'
        if _measure(word, vowels) == 1 and _ends_cvc(word, vowels):
            return word + "e"

    return word


def _step_1c(word: str) -> str:
    """Step 1c: Handle -iveness, -fulness, etc."""
    if word.endswith("ness") or word.endswith("ment") or word.endswith("fully"):
        # Don't stem these in step 1c
        pass

    # Handle -ive -> iv if m>1
    if word.endswith("ive"):
        stem = word[:-3]
        if _measure(stem) > 0:
            return stem

    return word


def _step_2(word: str) -> str:
    """Step 2: Handle various suffixes."""
    suffixes = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("fulli", "ful"),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 0:
                return stem + replacement

    return word


def _step_3(word: str) -> str:
    """Step 3: Handle -ic-, -full, -ness, etc."""
    suffixes = [
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 0:
                return stem + replacement

    return word


def _step_4(word: str) -> str:
    """Step 4: Handle -ive, -ment, etc."""
    suffixes = [
        ("al", ""),
        ("ance", ""),
        ("ence", ""),
        ("er", ""),
        ("ic", ""),
        ("able", ""),
        ("ible", ""),
        ("ate", ""),
        ("ive", ""),
        ("ize", ""),
        ("ment", ""),
        ("ant", ""),
        ("ent", ""),
        ("ism", ""),
        ("ou", ""),
        ("tion", ""),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 1:
                return stem + replacement

    return word


def _step_5a(word: str) -> str:
    """Step 5a: Handle -e endings."""
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1:
            return stem
        if m == 1 and not _ends_cvc(stem):
            return stem

    return word


def _step_5b(word: str) -> str:
    """Step 5b: Handle double consonants at the end."""
    m = _measure(word)
    if m > 1 and _ends_double_consonant(word) and _ends_cvc(word):
        return word[:-1]

    return word
