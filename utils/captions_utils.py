import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from config_loader import get_config

nltk.download('punkt')
nltk.download('stopwords')

class CaptionsPreprocessing:
    STOPWORDS = set(stopwords.words('english'))
    @staticmethod
    def preprocess_text(file):
        result = []
        for caption in file:
            caption = caption.lower()
            caption = re.sub(r'[^\w\s]', '', caption)

            caption = ' '.join(caption.split())
            result.append(caption)
        return result

    @staticmethod
    def tokenize_captions(file, remove_stopwords=True):
        result = []
        for caption in file:
            tokens = word_tokenize(caption)
            if remove_stopwords:
                tokens = [t for t in tokens if t not in CaptionsPreprocessing.STOPWORDS]
            result.append(' '.join(tokens))

        return result

    @staticmethod
    def build_vocab(file, min_freq=None, config_path=None):
        if min_freq is None:
            config = get_config(config_path)
            dataset_config = config.get_dataset_config()
            min_freq = dataset_config.get('min_word_freq', 2)

        word_freq = Counter()
        for caption in file:
            tokens = caption.split()
            word_freq.update(tokens)

        base_vocab = ['<START>', '<END>', '<UNK>', '<PAD>']
        for word, freq in word_freq.most_common():
            if freq >= min_freq:
                base_vocab.append(word)

        word2idx = {w: i for i, w in enumerate(base_vocab)}
        return word2idx, word_freq

    @staticmethod
    def add_special_tokens(file):
        return [f'<START> {caption} <END>' for caption in file]

    @classmethod
    def full_pipeline(cls, file, remove_stopwords=None, min_freq=None, config_path=None):
        if remove_stopwords is None or min_freq is None:
            config = get_config(config_path)
            dataset_config = config.get_dataset_config()
            if remove_stopwords is None:
                remove_stopwords = dataset_config.get('remove_stopwords', True)
            if min_freq is None:
                min_freq = dataset_config.get('min_word_freq', 2)

        preprocessed = cls.preprocess_text(file)
        with_tokens = cls.tokenize_captions(preprocessed, remove_stopwords=remove_stopwords)
        word2idx, word_freq = cls.build_vocab(with_tokens, min_freq=min_freq, config_path=config_path)
        with_spec = cls.add_special_tokens(with_tokens)

        return with_spec, word2idx, word_freq
