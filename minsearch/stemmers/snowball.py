"""
Snowball stemmer for English.

An English stemmer based on the Snowball stemming algorithm.
More aggressive than Porter, handles more edge cases.

This is a simplified implementation that covers common patterns.

Reference:
    The English (Porter2) stemming algorithm - Snowball
    https://snowballstem.org/algorithms/english/stemmer.html

    The algorithm follows these main steps:
    - Step 0: Remove apostrophes
    - Step 1a: Handle plurals (sses->ss, ies->i, s->delete)
    - Step 1b: Handle -ed, -ingly, -ing suffixes
    - Step 1c: Replace -y or -Y with -i if preceded by consonant
    - Step 2: Handle -tional, -enci, -anci, -abli, etc.
    - Step 3: Handle -ational, -tional, -alize, -icate, etc.
    - Step 4: Handle -al, -ance, -ence, -er, -ic, etc.
    - Step 5: Handle -e and -l endings

    Special cases handled:
    - skis->ski, skies->sky, sky->sky
    - idly->idl, gently->gentl, ugly->ugli, early->earli
    - dying->die, lying->lie, tying->tie
"""

from typing import Set


# Common consonant/vowel definitions
_vowels: Set[str] = {'a', 'e', 'i', 'o', 'u', 'y'}


def snowball_stemmer(word: str) -> str:
    """
    Apply the Snowball stemming algorithm to a word.

    This is a simplified implementation of the Snowball (Porter 2) stemmer.
    More aggressive than the original Porter stemmer.

    Algorithm specification:
        https://snowballstem.org/algorithms/english/stemmer.html

    Args:
        word: The word to stem.

    Returns:
        The stemmed word in lowercase.

    Examples:
        >>> snowball_stemmer("running")
        'run'
        >>> snowball_stemmer("generously")
        'gener'
        >>> snowball_stemmer("technical")
        'technic'
    """
    if not word:
        return ""

    word = word.lower()

    # Special common cases
    special_cases = {
        "skis": "ski",
        "skies": "sky",
        "sky": "sky",
        "dying": "die",
        "lying": "lie",
        "tying": "tie",
        "idly": "idl",
        "gently": "gentl",
        "gentle": "gentl",
        "ugly": "ugli",
        "early": "earli",
        "only": "onli",
        "singly": "singl",
    }
    if word in special_cases:
        return special_cases[word]

    # Step 1: Handle various suffixes
    word = _step_1(word)

    # Step 2: Handle more suffixes
    word = _step_2(word)

    # Step 3: Handle remaining suffixes
    word = _step_3(word)

    # Step 4: Handle final cleanup
    word = _step_4(word)

    return word


def _is_vowel(char: str) -> bool:
    """Check if a character is a vowel."""
    return char in _vowels


def _measure(word: str) -> int:
    """
    Calculate the measure of a word (number of consonant-vowel sequences).

    [C](VC)^m[V] where C is a consonant and V is a vowel.
    """
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = _is_vowel(char)
        if not is_vowel and prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    return count


def _step_1(word: str) -> str:
    """Step 1: Handle various suffixes."""
    # Plurals
    if word.endswith("sses"):
        return word[:-2]

    if word.endswith("ies"):
        return word[:-2]

    if word.endswith("ss"):
        return word

    if word.endswith("s"):
        return word[:-1]

    # Suffixes to search for in order
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


def _step_2(word: str) -> str:
    """Step 2: Handle more suffixes."""
    suffixes = [
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ling", "l"),
        ("fulli", "ful"),
        ("lessli", "less"),
        ("bl", "ble"),
        ("ogi", "og"),
        ("li", ""),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 0:
                if replacement:
                    return stem + replacement
                else:
                    # Special case for -li ending
                    if stem[-1] in _vowels:
                        return stem

    return word


def _step_3(word: str) -> str:
    """Step 3: Handle remaining suffixes."""
    # Suffixes that require measure > 0
    strict_suffixes = [
        ("ization", "ize"),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("tional", "tion"),
        ("biliti", "ble"),
    ]

    # Suffixes that only need a vowel in stem (more lenient)
    vowel_suffixes = [
        ("lessli", "less"),
        ("entli", "ent"),
        ("ation", "ate"),
        ("alism", "al"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("ousli", "ous"),
        ("fulli", "ful"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("ator", "ate"),
        ("ling", "l"),
        ("ness", ""),
        ("ment", ""),
    ]

    # Common suffixes that need vowel check
    common_suffixes = [
        ("ing", ""),
        ("ed", ""),
        ("ly", ""),
        ("able", ""),
        ("ible", ""),
        ("ic", ""),
        ("al", ""),
        ("er", ""),
        ("est", ""),
    ]

    # Check strict suffixes first
    for suffix, replacement in strict_suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 0:
                if replacement:
                    return stem + replacement
                else:
                    return stem

    # Check vowel suffixes
    for suffix, replacement in vowel_suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _contains_vowel(stem):
                if replacement:
                    return stem + replacement
                else:
                    return stem

    # Check common suffixes
    for suffix, replacement in common_suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _contains_vowel(stem):
                # Check for double consonant after removing common suffixes
                # This handles "running" -> "runn" -> "run"
                if _ends_double_consonant(stem):
                    if len(stem) >= 3:
                        stem = stem[:-1]

                if replacement:
                    return stem + replacement
                else:
                    return stem

    return word


def _contains_vowel(word: str) -> bool:
    """Check if a word contains a vowel."""
    return any(char in _vowels for char in word)


def _ends_double_consonant(word: str) -> bool:
    """Check if word ends with a double consonant."""
    if len(word) < 2:
        return False
    return word[-1] == word[-2] and word[-1] not in _vowels


def _step_4(word: str) -> str:
    """Step 4: Handle final cleanup."""
    suffixes = [
        ("e", ""),
        ("ement", ""),
        ("ance", ""),
        ("ence", ""),
        ("able", ""),
        ("ible", ""),
        ("ment", ""),
        ("ant", ""),
        ("ent", ""),
        ("ism", ""),
        ("ate", ""),
        ("iti", ""),
        ("ous", ""),
        ("ive", ""),
        ("ize", ""),
        ("ion", ""),
        ("al", ""),
        ("er", ""),
        ("est", ""),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 1:
                if replacement == "" and suffix == "ion":
                    # Special case for -ion: only remove if preceded by s or t
                    if stem and stem[-1] in {'s', 't'}:
                        return stem
                elif replacement:
                    return stem + replacement
                else:
                    return stem

    return word
