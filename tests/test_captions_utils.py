import nltk

from utils.captions_utils import CaptionsPreprocessing


def _ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


_ensure_nltk_data()


class TestCaptionsPreprocessing:
    def test_preprocess_text(self):
        example = ["HELLO, My NAME? IS"]
        prec = CaptionsPreprocessing.preprocess_text(example)
        assert prec[0] == "hello my name is"

    def test_remove_spaces(self):
        example = ["some      text"]
        prec = CaptionsPreprocessing.preprocess_text(example)
        assert prec[0] == "some text"

    def test_with_stopwords(self):
        example = ["me my myself"]
        prec = CaptionsPreprocessing.tokenize_captions(example, remove_stopwords=False)
        assert prec == example

    def test_without_stopwords(self):
        example = ["the cat is sleeping"]
        prec = CaptionsPreprocessing.tokenize_captions(example, remove_stopwords=True)
        assert "the" not in prec[0]
        assert "is" not in prec[0]
        assert "cat" in prec[0]

    def test_build_vocab_min_freq(self):
        example = ["dog dog dog", "cat cat", "bird"]
        word2idx, word_freq = CaptionsPreprocessing.build_vocab(example, min_freq=2)
        assert "dog" in word2idx
        assert "cat" in word2idx
        assert "bird" not in word2idx

    def test_build_vocab_word_freq(self):
        example = ["dog dog dog", "cat cat", "bird"]
        word2idx, word_freq = CaptionsPreprocessing.build_vocab(example, min_freq=2)
        assert word_freq["dog"] == 3
        assert word_freq["cat"] == 2

    def test_add_special_tokens(self):
        example = ["hello world"]
        with_special = CaptionsPreprocessing.add_special_tokens(example)
        assert with_special[0].startswith("<START>")
        assert with_special[0].endswith("<END>")
