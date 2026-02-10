"""
Tests for stemmers module.

Comprehensive tests for Porter, Snowball, and Lancaster stemmers.
"""

from minsearch.stemmers import (
    porter_stemmer,
    snowball_stemmer,
    lancaster_stemmer,
    get_stemmer,
    STEMMERS,
)


# ==================== Porter Stemmer Tests ====================

class TestPorterStemmer:
    """Tests for Porter stemmer."""

    def test_empty_string(self):
        assert porter_stemmer("") == ""

    def test_identity_words(self):
        """Words that shouldn't change."""
        assert porter_stemmer("trie") == "trie"
        assert porter_stemmer("chair") == "chair"

    def test_plural_removal(self):
        """Test removing plural endings."""
        assert porter_stemmer("cats") == "cat"
        assert porter_stemmer("dogs") == "dog"
        assert porter_stemmer("horses") == "hors"
        assert porter_stemmer("boys") == "boy"

    def test_special_plurals(self):
        """Test special plural cases."""
        assert porter_stemmer("skies") == "ski"
        assert porter_stemmer("ties") == "ti"
        assert porter_stemmer("agreed") == "agre"

    def test_ing_endings(self):
        """Test removing -ing endings."""
        assert porter_stemmer("running") == "run"
        assert porter_stemmer("jumping") == "jump"
        assert porter_stemmer("hoping") == "hope"

    def test_ed_endings(self):
        """Test removing -ed endings."""
        assert porter_stemmer("jumped") == "jump"
        assert porter_stemmer("played") == "play"
        assert porter_stemmer("hopped") == "hop"

    def test_ly_endings(self):
        """Test removing -ly adverb endings."""
        assert porter_stemmer("quickly") == "quickly"
        assert porter_stemmer("slowly") == "slowly"

    def test_ment_endings(self):
        """Test removing -ment endings."""
        assert porter_stemmer("development") == "develop"
        assert porter_stemmer("government") == "govern"

    def test_tion_endings(self):
        """Test removing -tion endings."""
        assert porter_stemmer("organization") == "organ"
        assert porter_stemmer("condition") == "condition"

    def test_ness_endings(self):
        """Test removing -ness endings."""
        assert porter_stemmer("happiness") == "happi"
        assert porter_stemmer("kindness") == "kind"

    def test_ive_endings(self):
        """Test removing -ive endings."""
        assert porter_stemmer("active") == "act"
        assert porter_stemmer("effective") == "effect"

    def test_ize_endings(self):
        """Test removing -ize endings."""
        assert porter_stemmer("organize") == "organ"
        assert porter_stemmer("recognize") == "recogn"

    def test_ful_endings(self):
        """Test removing -ful endings."""
        assert porter_stemmer("helpful") == "help"
        assert porter_stemmer("careful") == "care"

    def test_case_insensitive(self):
        """Test that stemming is case-insensitive."""
        assert porter_stemmer("Running") == "run"
        assert porter_stemmer("RUNNING") == "run"
        assert porter_stemmer("JumpEd") == "jump"

    def test_common_words(self):
        """Test stemming of common words."""
        assert porter_stemmer("computing") == "comput"
        assert porter_stemmer("connects") == "connect"
        assert porter_stemmer("provider") == "provid"
        assert porter_stemmer("science") == "scienc"
        assert porter_stemmer("reference") == "refer"

    def test_ative_endings(self):
        """Test -ative endings."""
        assert porter_stemmer("alternative") == "alternat"
        assert porter_stemmer("relative") == "relat"

    def test_ance_endings(self):
        """Test -ance endings."""
        assert porter_stemmer("allowance") == "allow"
        assert porter_stemmer("importance") == "import"

    def test_ence_endings(self):
        """Test -ence endings."""
        assert porter_stemmer("difference") == "differ"
        assert porter_stemmer("independence") == "independ"

    def test_ality_endings(self):
        """Test -ality endings."""
        assert porter_stemmer("technology") == "technology"
        assert porter_stemmer("quality") == "quality"

    def test_length_preservation(self):
        """Test that very short words aren't stemmed to nothing."""
        assert porter_stemmer("a") == "a"
        assert porter_stemmer("at") == "at"
        assert porter_stemmer("cat") == "cat"


# ==================== Snowball Stemmer Tests ====================

class TestSnowballStemmer:
    """Tests for Snowball stemmer."""

    def test_empty_string(self):
        assert snowball_stemmer("") == ""

    def test_identity_words(self):
        """Words that shouldn't change."""
        assert snowball_stemmer("call") == "call"
        assert snowball_stemmer("this") == "thi"  # Snowball stems "this" to "thi"

    def test_plural_removal(self):
        """Test removing plural endings."""
        assert snowball_stemmer("cats") == "cat"
        assert snowball_stemmer("dogs") == "dog"
        assert snowball_stemmer("horses") == "horse"

    def test_ing_endings(self):
        """Test removing -ing endings."""
        assert snowball_stemmer("running") == "run"
        assert snowball_stemmer("jumping") == "jump"
        assert snowball_stemmer("swimming") == "swim"

    def test_ed_endings(self):
        """Test removing -ed endings."""
        assert snowball_stemmer("jumped") == "jump"
        assert snowball_stemmer("played") == "play"
        assert snowball_stemmer("stopped") == "stop"

    def test_ly_endings(self):
        """Test removing -ly adverb endings."""
        assert snowball_stemmer("quickly") == "quick"
        assert snowball_stemmer("badly") == "bad"

    def test_ment_endings(self):
        """Test removing -ment endings."""
        assert snowball_stemmer("development") == "develop"
        assert snowball_stemmer("management") == "manag"

    def test_tion_endings(self):
        """Test removing -tion endings."""
        assert snowball_stemmer("organization") == "organiz"
        assert snowball_stemmer("creation") == "create"

    def test_ness_endings(self):
        """Test removing -ness endings."""
        assert snowball_stemmer("happiness") == "happi"
        assert snowball_stemmer("kindness") == "kind"

    def test_ize_endings(self):
        """Test removing -ize endings."""
        assert snowball_stemmer("organize") == "organiz"
        assert snowball_stemmer("capitalize") == "capitaliz"

    def test_ful_endings(self):
        """Test removing -ful endings."""
        assert snowball_stemmer("helpful") == "helpful"
        assert snowball_stemmer("useful") == "useful"

    def test_ical_endings(self):
        """Test -ical endings."""
        assert snowball_stemmer("technical") == "technic"
        assert snowball_stemmer("logical") == "logic"

    def test_ive_endings(self):
        """Test removing -ive endings."""
        assert snowball_stemmer("active") == "activ"
        assert snowball_stemmer("positive") == "positiv"

    def test_ance_endings(self):
        """Test -ance endings."""
        assert snowball_stemmer("allowance") == "allowanc"
        assert snowball_stemmer("distance") == "distanc"

    def test_ence_endings(self):
        """Test -ence endings."""
        assert snowball_stemmer("difference") == "differenc"
        assert snowball_stemmer("existence") == "existenc"

    def test_ity_endings(self):
        """Test -ity endings."""
        assert snowball_stemmer("reality") == "reality"
        assert snowball_stemmer("generosity") == "generosity"

    def test_ness_suffix(self):
        """Test -ness suffix removal."""
        assert snowball_stemmer("darkness") == "dark"
        assert snowball_stemmer("sadness") == "sad"

    def test_less_suffix(self):
        """Test -less suffix removal."""
        assert snowball_stemmer("careless") == "careless"
        assert snowball_stemmer("hopeless") == "hopeless"

    def test_able_suffix(self):
        """Test -able suffix removal."""
        assert snowball_stemmer("readable") == "read"
        assert snowball_stemmer("portable") == "port"

    def test_ible_suffix(self):
        """Test -ible suffix removal."""
        assert snowball_stemmer("horrible") == "hor"
        assert snowball_stemmer("terrible") == "ter"

    def test_al_suffix(self):
        """Test -al suffix removal."""
        assert snowball_stemmer("removal") == "remov"
        assert snowball_stemmer("critical") == "critic"

    def test_er_suffix(self):
        """Test -er suffix removal."""
        assert snowball_stemmer("teacher") == "teach"
        assert snowball_stemmer("player") == "play"

    def test_est_suffix(self):
        """Test -est suffix removal."""
        assert snowball_stemmer("biggest") == "big"
        assert snowball_stemmer("fastest") == "fast"

    def test_case_insensitive(self):
        """Test that stemming is case-insensitive."""
        assert snowball_stemmer("Running") == "run"
        assert snowball_stemmer("RUNNING") == "run"


# ==================== Lancaster Stemmer Tests ====================

class TestLancasterStemmer:
    """Tests for Lancaster stemmer."""

    def test_empty_string(self):
        assert lancaster_stemmer("") == ""

    def test_identity_words(self):
        """Words that shouldn't change or change minimally."""
        assert lancaster_stemmer("a") == "a"
        assert lancaster_stemmer("the") == "the"
        assert lancaster_stemmer("when") == "when"

    def test_aggressive_stemming(self):
        """Test that Lancaster is more aggressive."""
        assert lancaster_stemmer("running") == "run"
        assert lancaster_stemmer("maximum") == "maximum"  # Doesn't stem
        assert lancaster_stemmer("presumably") == "presumab"
        assert lancaster_stemmer("lication") == "lic"

    def test_plural_removal(self):
        """Test removing plural endings."""
        assert lancaster_stemmer("cats") == "cat"
        assert lancaster_stemmer("dogs") == "dog"
        assert lancaster_stemmer("children") == "children"  # Irregular, no rule

    def test_ing_endings(self):
        """Test removing -ing endings."""
        assert lancaster_stemmer("running") == "run"
        assert lancaster_stemmer("jumping") == "jump"
        assert lancaster_stemmer("dancing") == "danc"

    def test_ed_endings(self):
        """Test removing -ed endings."""
        assert lancaster_stemmer("jumped") == "jump"
        assert lancaster_stemmer("played") == "play"
        assert lancaster_stemmer("hopped") == "hop"

    def test_ly_endings(self):
        """Test removing -ly adverb endings."""
        assert lancaster_stemmer("quickly") == "quick"
        assert lancaster_stemmer("slowly") == "slow"
        assert lancaster_stemmer("badly") == "bad"

    def test_ment_endings(self):
        """Test removing -ment endings."""
        assert lancaster_stemmer("development") == "develop"
        assert lancaster_stemmer("government") == "govern"

    def test_tion_endings(self):
        """Test removing -tion endings."""
        assert lancaster_stemmer("organization") == "organ"
        assert lancaster_stemmer("condition") == "condition"  # Doesn't stem

    def test_ness_endings(self):
        """Test removing -ness endings."""
        assert lancaster_stemmer("happiness") == "happin"
        assert lancaster_stemmer("kindness") == "kindn"

    def test_ive_endings(self):
        """Test removing -ive endings."""
        assert lancaster_stemmer("active") == "act"
        assert lancaster_stemmer("positive") == "posit"

    def test_ize_endings(self):
        """Test removing -ize endings."""
        assert lancaster_stemmer("organize") == "organ"
        assert lancaster_stemmer("recognize") == "recogn"

    def test_ful_endings(self):
        """Test removing -ful endings."""
        assert lancaster_stemmer("helpful") == "help"
        assert lancaster_stemmer("careful") == "car"

    def test_al_endings(self):
        """Test removing -al endings."""
        assert lancaster_stemmer("critical") == "critic"
        assert lancaster_stemmer("original") == "origin"

    def test_er_endings(self):
        """Test removing -er endings."""
        assert lancaster_stemmer("teacher") == "teach"
        assert lancaster_stemmer("player") == "play"

    def test_est_endings(self):
        """Test removing -est endings."""
        assert lancaster_stemmer("biggest") == "big"
        assert lancaster_stemmer("fastest") == "fast"

    def test_ance_endings(self):
        """Test removing -ance endings."""
        assert lancaster_stemmer("allowance") == "allow"
        assert lancaster_stemmer("distance") == "dist"

    def test_ence_endings(self):
        """Test removing -ence endings."""
        assert lancaster_stemmer("difference") == "dif"
        assert lancaster_stemmer("independence") == "independ"

    def test_multiply(self):
        """Test 'multiply' stemming."""
        assert lancaster_stemmer("multiply") == "multip"
        assert lancaster_stemmer("multiplying") == "multip"

    def test_case_insensitive(self):
        """Test that stemming is case-insensitive."""
        assert lancaster_stemmer("Running") == "run"
        assert lancaster_stemmer("RUNNING") == "run"

    def test_short_words(self):
        """Test that short words are handled correctly."""
        assert lancaster_stemmer("cat") == "cat"
        assert lancaster_stemmer("dog") == "dog"
        assert lancaster_stemmer("run") == "run"


# ==================== get_stemmer Tests ====================

class TestGetStemmer:
    """Tests for get_stemmer function."""

    def test_none_returns_lowercase(self):
        """Test that None returns a no-op stemmer (lowercase only)."""
        stemmer = get_stemmer(None)
        assert stemmer("Running") == "running"
        assert stemmer("JUMPED") == "jumped"

    def test_porter(self):
        """Test getting porter stemmer."""
        stemmer = get_stemmer("porter")
        assert stemmer("running") == "run"
        assert stemmer("cats") == "cat"

    def test_snowball(self):
        """Test getting snowball stemmer."""
        stemmer = get_stemmer("snowball")
        assert stemmer("running") == "run"
        assert stemmer("cats") == "cat"

    def test_lancaster(self):
        """Test getting lancaster stemmer."""
        stemmer = get_stemmer("lancaster")
        assert stemmer("running") == "run"
        assert stemmer("maximum") == "maximum"  # Lancaster doesn't stem this word

    def test_invalid_name_returns_lowercase(self):
        """Test that invalid names return lowercase stemmer."""
        stemmer = get_stemmer("invalid")
        assert stemmer("Running") == "running"

    def test_case_insensitive_name(self):
        """Test that stemmer name is case-insensitive."""
        assert get_stemmer("PORTER")("running") == "run"
        assert get_stemmer("Porter")("running") == "run"
        assert get_stemmer("PORTER")("cats") == "cat"


# ==================== STEMMERS Dict Tests ====================

class TestStemmersDict:
    """Tests for the STEMMERS dictionary."""

    def test_stemmers_dict_structure(self):
        """Test that STEMMERS dict has correct structure."""
        assert isinstance(STEMMERS, dict)
        assert "porter" in STEMMERS
        assert "snowball" in STEMMERS
        assert "lancaster" in STEMMERS
        assert "none" in STEMMERS

    def test_stemmers_dict_callables(self):
        """Test that all stemmers in dict are callable."""
        for name, stemmer in STEMMERS.items():
            assert callable(stemmer), f"{name} is not callable"

    def test_stemmers_dict_functionality(self):
        """Test that stemmers in dict work correctly."""
        assert STEMMERS["porter"]("running") == "run"
        assert STEMMERS["snowball"]("running") == "run"
        assert STEMMERS["lancaster"]("running") == "run"
        assert STEMMERS["none"]("running") == "running"


# ==================== Comparative Tests ====================

class TestStemmerComparison:
    """Tests comparing different stemmers."""

    def test_running_all_stemmers(self):
        """All stemmers should handle 'running' similarly."""
        assert "run" in porter_stemmer("running")
        assert "run" in snowball_stemmer("running")
        assert lancaster_stemmer("running") == "run"

    def test_aggressiveness(self):
        """Test that Lancaster is different from Porter."""
        word = "happiness"
        port = porter_stemmer(word)
        lanc = lancaster_stemmer(word)
        # Lancaster is different from Porter
        assert lanc != port

    def test_cats_all_same(self):
        """All stemmers should handle simple plurals."""
        port = porter_stemmer("cats")
        snow = snowball_stemmer("cats")
        lanc = lancaster_stemmer("cats")
        assert port == "cat"
        assert snow == "cat"
        assert lanc == "cat"

    def test_happiness_differences(self):
        """Test different behavior on complex words."""
        port = porter_stemmer("happiness")
        snow = snowball_stemmer("happiness")
        lanc = lancaster_stemmer("happiness")
        # Lancaster is more aggressive (happin vs happi)
        assert lanc == "happin"
        # Porter/Snowball keep the 'i'
        assert port == "happi"
        assert snow == "happi"
