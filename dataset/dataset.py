import os
import random
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset

from config_loader import get_config
from utils.captions_utils import CaptionsPreprocessing
from utils.images_utils import get_image_by_name, make_transform

SPECIAL_TOKENS = ("<START>", "<END>", "<UNK>", "<PAD>")


def split_image_names(image_names, train_size=0.8, val_size=0.1, seed=42):
    if train_size < 0 or val_size < 0 or train_size + val_size > 1:
        raise ValueError("train_size and val_size must be non-negative and sum to <= 1")

    image_names = list(image_names)
    random.Random(seed).shuffle(image_names)

    train_length = int(train_size * len(image_names))
    val_length = int(val_size * len(image_names))

    train_images = image_names[:train_length]
    val_images = image_names[train_length : train_length + val_length]
    test_images = image_names[train_length + val_length :]
    return train_images, val_images, test_images


class FlickrDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        images_path=None,
        captions_file=None,
        is_train=True,
        max_caption_length=None,
        min_word_freq=None,
        remove_stopwords=None,
        config_path=None,
        allowed_images=None,
        word2idx=None,
        word_freq=None,
    ):
        dataset_config = get_config(config_path).get_dataset_config()

        self.data_path = data_path or dataset_config.get("data_path", "./data")
        self.images_path = images_path or dataset_config.get("images_path", "images")
        self.max_caption_length = max_caption_length or dataset_config.get(
            "max_caption_length", 50
        )
        self.min_word_freq = min_word_freq or dataset_config.get("min_word_freq", 2)
        self.remove_stopwords = (
            remove_stopwords
            if remove_stopwords is not None
            else dataset_config.get("remove_stopwords", True)
        )
        self.allowed_images = (
            set(allowed_images) if allowed_images is not None else None
        )
        self.external_word2idx = word2idx
        self.external_word_freq = word_freq

        captions_file = captions_file or dataset_config.get(
            "captions_file", "captions.txt"
        )
        captions_path = os.path.join(self.data_path, captions_file)
        self.captions_df = pd.read_csv(captions_path)

        self.transform = make_transform(is_train)
        self._prepare_captions()
        self._create_items()

    def _prepare_captions(self):
        image_captions = defaultdict(list)
        for _, row in self.captions_df.iterrows():
            image_captions[row["image"]].append(row["caption"])

        selected_image_names = [
            image_name
            for image_name in image_captions
            if self.allowed_images is None or image_name in self.allowed_images
        ]

        self.image_names = selected_image_names
        self.processed_image_captions = defaultdict(list)
        selected_raw_captions = []

        for image_name in selected_image_names:
            raw_captions = image_captions[image_name]
            processed_captions = CaptionsPreprocessing.preprocess_text(raw_captions)
            tokenized_captions = CaptionsPreprocessing.tokenize_captions(
                processed_captions,
                remove_stopwords=self.remove_stopwords,
            )
            self.processed_image_captions[image_name] = (
                CaptionsPreprocessing.add_special_tokens(tokenized_captions)
            )
            selected_raw_captions.extend(raw_captions)

        if self.external_word2idx is not None:
            self.word2idx = self.external_word2idx
            self.word_freq = self.external_word_freq or {}
        else:
            processed_captions = CaptionsPreprocessing.preprocess_text(
                selected_raw_captions
            )
            tokenized_captions = CaptionsPreprocessing.tokenize_captions(
                processed_captions,
                remove_stopwords=self.remove_stopwords,
            )
            self.word2idx, self.word_freq = CaptionsPreprocessing.build_vocab(
                tokenized_captions,
                min_freq=self.min_word_freq,
            )

        self.idx2word = {index: word for word, index in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.start_idx, self.end_idx, self.unk_idx, self.pad_idx = (
            self.word2idx[token] for token in SPECIAL_TOKENS
        )

    def _create_items(self):
        self.dataset_items = []
        for image_name in self.image_names:
            for caption in self.processed_image_captions[image_name]:
                self.dataset_items.append((image_name, caption))

    def _caption_to_indices(self, caption):
        return [self.word2idx.get(token, self.unk_idx) for token in caption.split()]

    def _pad_caption(self, caption_indices):
        trimmed = caption_indices[: self.max_caption_length]
        padding = [self.pad_idx] * (self.max_caption_length - len(trimmed))
        return trimmed + padding

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, index):
        image_name, caption_text = self.dataset_items[index]
        image = get_image_by_name(
            self.data_path,
            self.images_path,
            image_name,
            transform=self.transform,
        )

        caption_indices = self._caption_to_indices(caption_text)
        return {
            "image": image,
            "caption": torch.tensor(
                self._pad_caption(caption_indices),
                dtype=torch.long,
            ),
            "caption_length": len(caption_indices),
        }

    def get_vocab_size(self):
        return self.vocab_size

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word


def collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "captions": torch.stack([item["caption"] for item in batch]),
        "caption_lengths": torch.tensor(
            [item["caption_length"] for item in batch],
            dtype=torch.long,
        ),
    }


class FlickrMetricsDataset(FlickrDataset):
    def _create_items(self):
        self.dataset_items = [
            (image_name, captions)
            for image_name, captions in self.processed_image_captions.items()
        ]

    def __getitem__(self, idx):
        image_name, captions = self.dataset_items[idx]
        image = get_image_by_name(
            self.data_path, self.images_path, image_name, transform=self.transform
        )

        references = [self._caption_to_indices(caption) for caption in captions]

        return {"image": image, "references": references}


def metrics_collate_fn(batch):
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "references": [item["references"] for item in batch],
    }
