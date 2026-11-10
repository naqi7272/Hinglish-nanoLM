from hnlm.data.clean import CleanConfig, clean_document, normalize, script_label, script_stats

CFG = CleanConfig()


def test_tweet_noise_is_removed():
    raw = "RT @dost: Aaj ka match &amp; party 🎉🎉 #mazaa https://t.co/xyz sooooo good yaar"
    text, script = clean_document(raw, CFG)
    assert text == "Aaj ka match & party mazaa sooo good yaar"
    assert script == "latn"


def test_devanagari_vowel_signs_survive():
    # Vowel signs (ि, ी, ं) are combining marks; they must not be stripped or split off.
    text, script = clean_document("मुझे हिंदी किताबें पढ़ना पसंद है", CFG)
    assert text == "मुझे हिंदी किताबें पढ़ना पसंद है"
    assert script == "deva"


def test_zwj_and_zwnj_are_kept_but_zero_width_space_is_not():
    assert normalize("क्‍ष a​b", CFG) == "क्‍ष ab"


def test_mixed_script_label():
    assert script_label(script_stats("मैं कल Delhi जा रहा हूँ for a meeting")) == "mixed"


def test_filters_drop_junk():
    assert clean_document("hi", CFG) is None  # too short
    assert clean_document("12345 67890 11121 31415 92653", CFG) is None  # mostly digits
    assert clean_document("这是一个中文句子，不是我们的目标语言。", CFG) is None  # other script
