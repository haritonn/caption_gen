import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from collections import defaultdict

from utils.captions_utils import CaptionsPreprocessing
from utils.images_utils import get_image_by_name, make_transform
from config_loader import get_config


class FlickrDataset(Dataset):
    def __init__(self, data_path=None, images_path=None, captions_file=None,
                 is_train=True, max_caption_length=None, min_word_freq=None,
                 remove_stopwords=None, config_path=None):

        config = get_config(config_path)
        dataset_config = config.get_dataset_config()

        self.data_path = data_path or dataset_config.get('data_path', './data')
        self.images_path = images_path or dataset_config.get('images_path', 'images')
        self.is_train = is_train
        self.max_caption_length = max_caption_length or dataset_config.get('max_caption_length', 50)
        self.min_word_freq = min_word_freq or dataset_config.get('min_word_freq', 2)
        self.remove_stopwords = remove_stopwords if remove_stopwords is not None else dataset_config.get('remove_stopwords', True)

        captions_file = captions_file or dataset_config.get('captions_file', 'captions.txt')

        self.captions_df = pd.read_csv(os.path.join(self.data_path, captions_file))
        self._prepare_captions()

        self.transform = make_transform(is_train)

        # Create mappings
        self._create_mappings()

    def _prepare_captions(self):
        self.image_captions = defaultdict(list)
        for _, row in self.captions_df.iterrows():
            self.image_captions[row['image']].append(row['caption'])

        self.image_names = list(self.image_captions.keys())

        all_captions = []
        for captions in self.image_captions.values():
            all_captions.extend(captions)

        self.processed_captions, self.word2idx, self.word_freq = CaptionsPreprocessing.full_pipeline(
            all_captions,
            remove_stopwords=self.remove_stopwords,
            min_freq=self.min_word_freq
        )

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        self.start_idx = self.word2idx['<START>']
        self.end_idx = self.word2idx['<END>']
        self.unk_idx = self.word2idx['<UNK>']
        self.pad_idx = self.word2idx['<PAD>']

    def _create_mappings(self):
        self.processed_image_captions = defaultdict(list)
        caption_idx = 0

        for image_name in self.image_names:
            num_captions = len(self.image_captions[image_name])
            for _ in range(num_captions):
                self.processed_image_captions[image_name].append(
                    self.processed_captions[caption_idx]
                )
                caption_idx += 1

        self.dataset_items = []
        for image_name in self.image_names:
            for caption in self.processed_image_captions[image_name]:
                self.dataset_items.append((image_name, caption))

    def _caption_to_indices(self, caption):
        tokens = caption.split()
        indices = []

        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.unk_idx)

        return indices

    def _pad_caption(self, caption_indices):
        if len(caption_indices) > self.max_caption_length:
            caption_indices = caption_indices[:self.max_caption_length]
        else:
            caption_indices.extend([self.pad_idx] * (self.max_caption_length - len(caption_indices)))

        return caption_indices

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, idx):
        image_name, caption_text = self.dataset_items[idx]

        try:
            image = get_image_by_name(
                self.data_path,
                self.images_path,
                image_name,
                transform=self.transform
            )
        except Exception as e:
            print(f"Warning: Could not load image {image_name}. Using dummy image. Error: {e}")
            dummy_image = Image.new('RGB', (224, 224), color='black')
            image = self.transform(dummy_image) if self.transform else dummy_image

        caption_indices = self._caption_to_indices(caption_text)
        caption_length = len(caption_indices)

        padded_caption = self._pad_caption(caption_indices.copy())
        return {
            'image': image,
            'caption': torch.tensor(padded_caption, dtype=torch.long),
            'caption_length': caption_length,
            'image_name': image_name,
            'caption_text': caption_text
        }

    def get_vocab_size(self):
        return self.vocab_size

    def get_word2idx(self):
        return self.word2idx

    def get_idx2word(self):
        return self.idx2word

    def decode_caption(self, caption_indices):
        if torch.is_tensor(caption_indices):
            caption_indices = caption_indices.tolist()

        words = []
        for idx in caption_indices:
            if idx == self.pad_idx:
                break
            if idx in self.idx2word:
                words.append(self.idx2word[idx])

        return ' '.join(words)

    def get_image_captions_count(self, image_name):
        return len(self.image_captions.get(image_name, []))

    def get_all_captions_for_image(self, image_name):
        return self.image_captions.get(image_name, [])


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = torch.stack([item['caption'] for item in batch])
    caption_lengths = torch.tensor([item['caption_length'] for item in batch], dtype=torch.long)
    image_names = [item['image_name'] for item in batch]
    caption_texts = [item['caption_text'] for item in batch]

    return {
        'images': images,
        'captions': captions,
        'caption_lengths': caption_lengths,
        'image_names': image_names,
        'caption_texts': caption_texts
    }
