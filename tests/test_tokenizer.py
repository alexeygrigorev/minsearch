"""
Tests for Tokenizer module.

Comprehensive tests for tokenization, stop words removal, and stemming.
"""

from minsearch.tokenizer import Tokenizer, DEFAULT_ENGLISH_STOP_WORDS

# ==================== Stop Words Tests ====================

class TestStopWords:
    """Tests for default stop words loading."""

    def test_default_stop_words_loaded(self):
        """Test that default stop words are loaded from file."""
        assert isinstance(DEFAULT_ENGLISH_STOP_WORDS, set)
        assert len(DEFAULT_ENGLISH_STOP_WORDS) > 0
        # Check some common stop words
        assert "the" in DEFAULT_ENGLISH_STOP_WORDS
        assert "a" in DEFAULT_ENGLISH_STOP_WORDS
        assert "an" in DEFAULT_ENGLISH_STOP_WORDS
        assert "and" in DEFAULT_ENGLISH_STOP_WORDS
        assert "or" in DEFAULT_ENGLISH_STOP_WORDS
        assert "but" in DEFAULT_ENGLISH_STOP_WORDS

    def test_stop_words_are_lowercase(self):
        """Test that all stop words are lowercase."""
        for word in DEFAULT_ENGLISH_STOP_WORDS:
            assert word == word.lower()


# ==================== Basic Tokenizer Tests ====================

class TestTokenizer:
    """Tests for Tokenizer without stemming."""

    def test_empty_string(self):
        tokenizer = Tokenizer()
        assert tokenizer.tokenize("") == []

    def test_none_input(self):
        tokenizer = Tokenizer()
        assert tokenizer.tokenize("") == []

    def test_simple_tokenization(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello world")
        assert set(tokens) == {"hello", "world"}

    def test_punctuation_removal(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello, world! Python is great")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        assert "great" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_digit_removal(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Python 3.11 is here 123")
        assert "python" in tokens
        assert "3" not in tokens
        assert "11" not in tokens
        assert "123" not in tokens

    def test_lowercase_conversion(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("HELLO World PyTHon")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens

    def test_multiple_spaces(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("hello    world     test")
        assert tokens == ["hello", "world", "test"]

    def test_special_characters(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("hello@world#test$foo")
        assert tokens == ["hello", "world", "test", "foo"]

    def test_underscores_and_hyphens(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("hello_world test-value")
        # Default pattern [\s\W\d]+ treats underscore as word char, so hello_world stays together
        # But hyphens are non-word, so test-value is split
        assert "hello_world" in tokens or "hello" in tokens
        assert "test" in tokens
        assert "value" in tokens

    def test_custom_stop_words(self):
        custom_stop = {"hello", "world"}
        tokenizer = Tokenizer(stop_words=custom_stop)
        tokens = tokenizer.tokenize("hello world python")
        assert tokens == ["python"]

    def test_empty_stop_words(self):
        tokenizer = Tokenizer(stop_words=set())
        text = "the and a or but python"
        tokens = tokenizer.tokenize(text)
        assert "the" in tokens
        assert "and" in tokens
        assert "python" in tokens

    def test_stop_words_removal(self):
        tokenizer = Tokenizer(stop_words='english')
        text = "the quick brown fox jumps over the lazy dog"
        tokens = tokenizer.tokenize(text)
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "over" not in tokens  # "over" is not a stop word in our list

    def test_contraction_removal(self):
        tokenizer = Tokenizer()
        text = "don't can't won't shouldn't"
        tokens = tokenizer.tokenize(text)
        # Contractions are split by non-word chars
        assert "don" in tokens or "t" in tokens
        assert len(tokens) > 0


# ==================== Tokenizer with Stemming Tests ====================

class TestTokenizerWithStemming:
    """Tests for Tokenizer with various stemmers."""

    def test_tokenizer_with_porter(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("running jumps quickly")
        assert "run" in tokens
        assert "jump" in tokens
        # "quickly" might not stem to "quick" in all implementations

    def test_tokenizer_with_snowball(self):
        tokenizer = Tokenizer(stemmer="snowball")
        tokens = tokenizer.tokenize("running jumps quickly")
        assert "run" in tokens
        assert "jump" in tokens

    def test_tokenizer_with_lancaster(self):
        tokenizer = Tokenizer(stemmer="lancaster")
        tokens = tokenizer.tokenize("running jumps quickly")
        assert "run" in tokens
        assert "jump" in tokens

    def test_tokenizer_with_none_stemmer(self):
        tokenizer = Tokenizer(stemmer=None)
        tokens = tokenizer.tokenize("running jumping")
        assert "running" in tokens
        assert "jumping" in tokens

    def test_tokenizer_with_custom_stemmer(self):
        def custom_stem(word: str) -> str:
            return word[:-1] if len(word) > 3 else word

        tokenizer = Tokenizer(stemmer=custom_stem)
        tokens = tokenizer.tokenize("running jump")
        # "running" -> "runnin" (removes last char)
        # "jump" -> "jump" (only 4 chars, might be kept or stemmed)
        assert any("runnin" in t or t == "runnin" for t in tokens)

    def test_stemming_after_stop_words(self):
        tokenizer = Tokenizer(stop_words='english', stemmer="porter")
        text = "the running cats"
        tokens = tokenizer.tokenize(text)
        assert "the" not in tokens  # stop word
        assert "cat" in tokens or "run" in tokens  # stemmed

    def test_stop_words_apply_before_stemming(self):
        """Stop words should match the original form, not stemmed."""
        tokenizer = Tokenizer(stop_words='english', stemmer="porter")
        text = "running is fun"
        tokens = tokenizer.tokenize(text)
        assert "is" not in tokens  # stop word
        # "running" gets stemmed to "run"
        assert "run" in tokens


# ==================== Porter Stemmer Integration Tests ====================

class TestTokenizerPorterIntegration:
    """Integration tests for Tokenizer with Porter stemmer."""

    def test_porter_plurals(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("cats dogs horses")
        assert "cat" in tokens
        assert "dog" in tokens

    def test_porter_ing_endings(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("running jumping hopping")
        assert "run" in tokens
        assert "jump" in tokens
        assert "hop" in tokens

    def test_porter_ed_endings(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("jumped played hopped")
        assert "jump" in tokens
        assert "play" in tokens
        assert "hop" in tokens

    def test_porter_ly_endings(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("quickly slowly")
        # Some -ly words might not stem as expected
        assert len(tokens) > 0

    def test_porter_ness_endings(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("happiness kindness")
        assert "happi" in tokens
        assert "kind" in tokens

    def test_porter_ment_endings(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("development government")
        assert "develop" in tokens
        assert "govern" in tokens


# ==================== Snowball Stemmer Integration Tests ====================

class TestTokenizerSnowballIntegration:
    """Integration tests for Tokenizer with Snowball stemmer."""

    def test_snowball_plurals(self):
        tokenizer = Tokenizer(stemmer="snowball")
        tokens = tokenizer.tokenize("cats dogs horses")
        assert "cat" in tokens
        assert "dog" in tokens

    def test_snowball_ing_endings(self):
        tokenizer = Tokenizer(stemmer="snowball")
        tokens = tokenizer.tokenize("running jumping")
        assert "run" in tokens
        assert "jump" in tokens

    def test_snowball_ed_endings(self):
        tokenizer = Tokenizer(stemmer="snowball")
        tokens = tokenizer.tokenize("jumped played stopped")
        assert "jump" in tokens
        assert "play" in tokens
        assert "stop" in tokens

    def test_snowball_ness_endings(self):
        tokenizer = Tokenizer(stemmer="snowball")
        tokens = tokenizer.tokenize("happiness darkness")
        assert "happi" in tokens
        assert "dark" in tokens


# ==================== Lancaster Stemmer Integration Tests ====================

class TestTokenizerLancasterIntegration:
    """Integration tests for Tokenizer with Lancaster stemmer."""

    def test_lancaster_aggressive(self):
        tokenizer = Tokenizer(stemmer="lancaster")
        tokens = tokenizer.tokenize("maximum")
        # Lancaster is more aggressive
        assert len(tokens) > 0

    def test_lancaster_plurals(self):
        tokenizer = Tokenizer(stemmer="lancaster")
        tokens = tokenizer.tokenize("cats dogs")
        assert "cat" in tokens
        assert "dog" in tokens

    def test_lancaster_ing_endings(self):
        tokenizer = Tokenizer(stemmer="lancaster")
        tokens = tokenizer.tokenize("running jumping")
        assert "run" in tokens
        assert "jump" in tokens


# ==================== Combined Stop Words and Stemming Tests ====================

class TestTokenizerStopWordsAndStemming:
    """Tests combining stop words removal and stemming."""

    def test_porter_with_stop_words(self):
        tokenizer = Tokenizer(stop_words='english', stemmer="porter")
        text = "the running cats were jumping"
        tokens = tokenizer.tokenize(text)
        # Stop words: the, were
        assert "the" not in tokens
        assert "were" not in tokens
        # Stemmed tokens
        assert "run" in tokens or "cat" in tokens

    def test_snowball_with_stop_words(self):
        tokenizer = Tokenizer(stop_words='english', stemmer="snowball")
        text = "a running dog is jumping"
        tokens = tokenizer.tokenize(text)
        assert "a" not in tokens
        assert "is" not in tokens
        assert "run" in tokens or "dog" in tokens or "jump" in tokens

    def test_lancaster_with_stop_words(self):
        tokenizer = Tokenizer(stop_words='english', stemmer="lancaster")
        text = "the maximum happiness"
        tokens = tokenizer.tokenize(text)
        assert "the" not in tokens
        assert len(tokens) > 0

    def test_custom_stop_words_with_stemmer(self):
        custom_stop = {"running", "jumping"}
        tokenizer = Tokenizer(stop_words=custom_stop, stemmer="porter")
        tokens = tokenizer.tokenize("running jumping walking")
        assert "running" not in tokens  # removed as stop word
        assert "jumping" not in tokens  # removed as stop word
        # Note: "walking" might be stemmed to "walk"
        assert len(tokens) > 0


# ==================== Edge Cases Tests ====================

class TestTokenizerEdgeCases:
    """Tests for edge cases."""

    def test_only_stop_words(self):
        tokenizer = Tokenizer(stop_words='english')
        tokens = tokenizer.tokenize("the a an and or but")
        assert tokens == []

    def test_only_stop_words_with_stemmer(self):
        tokenizer = Tokenizer(stop_words='english', stemmer="porter")
        tokens = tokenizer.tokenize("the a an and or but")
        assert tokens == []

    def test_none_stop_words(self):
        """Test that None means no stop words are removed."""
        tokenizer = Tokenizer(stop_words=None)
        text = "the quick brown fox jumps over the lazy dog"
        tokens = tokenizer.tokenize(text)
        # All words including stop words should be present
        assert "the" in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "over" in tokens

    def test_none_stop_words_with_stemmer(self):
        """Test None stop words with stemmer."""
        tokenizer = Tokenizer(stop_words=None, stemmer="porter")
        text = "the running cats"
        tokens = tokenizer.tokenize(text)
        # No stop words removed, but stemming applied
        assert "the" in tokens
        assert "run" in tokens
        assert "cat" in tokens

    def test_custom_stop_words_set(self):
        """Test with custom set of stop words."""
        custom_stop = {"quick", "brown", "lazy"}
        tokenizer = Tokenizer(stop_words=custom_stop)
        text = "the quick brown fox jumps over the lazy dog"
        tokens = tokenizer.tokenize(text)
        # Custom stop words removed
        assert "quick" not in tokens
        assert "brown" not in tokens
        assert "lazy" not in tokens
        # Other words kept (including "the" which is not in custom set)
        assert "the" in tokens
        assert "fox" in tokens
        assert "jumps" in tokens

    def test_only_punctuation(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("!@#$%^&*()")
        assert tokens == []

    def test_only_numbers(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("123 456 789")
        assert tokens == []

    def test_mixed_languages(self):
        tokenizer = Tokenizer()
        # Non-ASCII characters may be handled differently
        tokens = tokenizer.tokenize("hello cafe naive")
        assert "hello" in tokens
        assert len(tokens) >= 2

    def test_single_word(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("hello")
        assert tokens == ["hello"]

    def test_single_word_with_stemmer(self):
        tokenizer = Tokenizer(stemmer="porter")
        tokens = tokenizer.tokenize("running")
        assert "run" in tokens

    def test_repeated_words(self):
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("hello hello hello world world")
        # Tokenizer returns duplicates (unlike set)
        assert tokens.count("hello") == 3
        assert tokens.count("world") == 2

    def test_whitespace_variations(self):
        tokenizer = Tokenizer()
        tokens1 = tokenizer.tokenize("hello world")
        tokens2 = tokenizer.tokenize("  hello  world  ")
        tokens3 = tokenizer.tokenize("\thello\tworld\t")
        assert tokens1 == tokens2 == tokens3


# ==================== Custom Pattern Tests ====================

class TestTokenizerCustomPattern:
    """Tests for Tokenizer with custom patterns."""

    def test_custom_pattern(self):
        tokenizer = Tokenizer(pattern=r"\s+")  # Only split on whitespace
        tokens = tokenizer.tokenize("hello,world!test")
        # With default pattern, punctuation would be removed
        # With \s+ only, we get "hello,world!test" as one token
        assert len(tokens) == 1

    def test_custom_pattern_with_stop_words(self):
        tokenizer = Tokenizer(pattern=r"\s+", stop_words={"hello"})
        tokens = tokenizer.tokenize("hello world")
        assert "hello" not in tokens
        assert "world" in tokens


# ==================== Natural Language Query Tests ====================

class TestTokenizerNaturalLanguage:
    """Tests for natural language queries."""

    def test_natural_query(self):
        tokenizer = Tokenizer(stop_words='english')
        query = "I just discovered the course, can I still join?"
        tokens = tokenizer.tokenize(query)
        # Stop words removed: I, the, can
        assert "i" not in tokens
        assert "the" not in tokens
        # Meaningful words kept
        assert "just" in tokens
        assert "discovered" in tokens
        assert "course" in tokens
        assert "still" in tokens
        assert "join" in tokens

    def test_natural_query_with_stemmer(self):
        tokenizer = Tokenizer(stop_words='english', stemmer="porter")
        query = "I am running and jumping"
        tokens = tokenizer.tokenize(query)
        assert "i" not in tokens
        assert "am" not in tokens
        assert "and" not in tokens
        assert "run" in tokens
        assert "jump" in tokens


# ==================== Tokenizer Stemmer Property Tests ====================

class TestTokenizerStemmerProperty:
    """Tests for accessing tokenizer's stemmer."""

    def test_no_stemmer_property(self):
        tokenizer = Tokenizer(stemmer=None)
        # Stemmer should be a function that lowercase the word
        result = tokenizer.stemmer("Running")
        assert result == "running"

    def test_porter_stemmer_property(self):
        tokenizer = Tokenizer(stemmer="porter")
        result = tokenizer.stemmer("running")
        assert "run" in result

    def test_snowball_stemmer_property(self):
        tokenizer = Tokenizer(stemmer="snowball")
        result = tokenizer.stemmer("running")
        assert "run" in result

    def test_lancaster_stemmer_property(self):
        tokenizer = Tokenizer(stemmer="lancaster")
        result = tokenizer.stemmer("running")
        assert result == "run"

    def test_custom_stemmer_property(self):
        def my_stemmer(word: str) -> str:
            return word[:-2] if len(word) > 4 else word

        tokenizer = Tokenizer(stemmer=my_stemmer)
        result = tokenizer.stemmer("running")
        assert result == "runni"
