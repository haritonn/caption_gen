import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config_loader import get_config


for resource_path, resource_name in (
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)


class CaptionsPreprocessing:
    STOPWORDS = set(stopwords.words("english"))

    @staticmethod
    def preprocess_text(captions):
        result = []
        for caption in captions:
            normalized = caption.lower()
            normalized = re.sub(r"[^\w\s]", "", normalized)
            normalized = " ".join(normalized.split())
            result.append(normalized)
        return result

    @staticmethod
    def tokenize_captions(captions, remove_stopwords=True):
        result = []
        for caption in captions:
            tokens = word_tokenize(caption)
            if remove_stopwords:
                tokens = [token for token in tokens if token not in CaptionsPreprocessing.STOPWORDS]
            result.append(" ".join(tokens))
        return result

    @staticmethod
    def build_vocab(captions, min_freq=None, config_path=None):
        if min_freq is None:
            min_freq = get_config(config_path).get_dataset_config().get("min_word_freq", 2)

        word_freq = Counter()
        for caption in captions:
            word_freq.update(caption.split())

        vocab = ["<START>", "<END>", "<UNK>", "<PAD>"]
        for word, freq in word_freq.most_common():
            if freq >= min_freq:
                vocab.append(word)

        word2idx = {word: index for index, word in enumerate(vocab)}
        return word2idx, word_freq

    @staticmethod
    def add_special_tokens(captions):
        return [f"<START> {caption} <END>" for caption in captions]
