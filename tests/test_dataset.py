import pytest
import torch
import os
import tempfile
import pandas as pd
from PIL import Image
import shutil
from dataset.dataset import FlickrDataset, collate_fn
from config_loader import get_config

class TestFlickrDataset:
    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()

        sample_data = {
            'image': [
                'test_image1.jpg', 'test_image1.jpg', 'test_image1.jpg',
                'test_image2.jpg', 'test_image2.jpg',
                'test_image3.jpg'
            ],
            'caption': [
                'A red car on the road',
                'Red vehicle driving on street',
                'Car moving on the road',
                'A dog playing in the park',
                'Happy dog running in grass',
                'Beautiful sunset over the mountains'
            ]
        }

        df = pd.DataFrame(sample_data)
        captions_path = os.path.join(temp_dir, 'captions.txt')
        df.to_csv(captions_path, index=False)

        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir)

        for image_name in sample_data['image']:
            if not os.path.exists(os.path.join(images_dir, image_name)):
                dummy_image = Image.new('RGB', (224, 224), color='red')
                dummy_image.save(os.path.join(images_dir, image_name))

        yield temp_dir

        shutil.rmtree(temp_dir)

    def test_dataset_initialization(self, temp_data_dir):
        """Test dataset initialization."""
        dataset = FlickrDataset(data_path=temp_data_dir, is_train=True)

        assert len(dataset) == 6
        assert dataset.get_vocab_size() > 0
        assert len(dataset.image_names) == 3

        word2idx = dataset.get_word2idx()
        assert '<START>' in word2idx
        assert '<END>' in word2idx
        assert '<UNK>' in word2idx
        assert '<PAD>' in word2idx

    def test_dataset_getitem(self, temp_data_dir):
        dataset = FlickrDataset(data_path=temp_data_dir, max_caption_length=20)

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert 'image' in sample
        assert 'caption' in sample
        assert 'caption_length' in sample
        assert 'image_name' in sample
        assert 'caption_text' in sample

        assert torch.is_tensor(sample['image'])
        assert torch.is_tensor(sample['caption'])
        assert sample['image'].shape == (3, 224, 224)
        assert sample['caption'].shape == (20,)
        assert isinstance(sample['caption_length'], int)
        assert isinstance(sample['image_name'], str)
        assert isinstance(sample['caption_text'], str)

    def test_caption_preprocessing(self, temp_data_dir):
        dataset = FlickrDataset(data_path=temp_data_dir)

        sample = dataset[0]
        original_caption = sample['caption_text']
        caption_tensor = sample['caption']
        decoded_caption = dataset.decode_caption(caption_tensor)

        assert '<START>' in decoded_caption
        assert '<END>' in decoded_caption

        indices = dataset._caption_to_indices(original_caption)
        assert isinstance(indices, list)
        assert all(isinstance(idx, int) for idx in indices)

    def test_vocabulary_methods(self, temp_data_dir):
        dataset = FlickrDataset(data_path=temp_data_dir, min_word_freq=1)

        word2idx = dataset.get_word2idx()
        idx2word = dataset.get_idx2word()
        vocab_size = dataset.get_vocab_size()

        assert isinstance(word2idx, dict)
        assert isinstance(idx2word, dict)
        assert isinstance(vocab_size, int)
        assert len(word2idx) == len(idx2word) == vocab_size

        for word, idx in word2idx.items():
            assert idx2word[idx] == word

    def test_caption_padding(self, temp_data_dir):
        max_length = 10
        dataset = FlickrDataset(data_path=temp_data_dir, max_caption_length=max_length)

        sample = dataset[0]
        caption_tensor = sample['caption']
        assert caption_tensor.shape[0] == max_length

        short_indices = [1, 2, 3]
        padded = dataset._pad_caption(short_indices.copy())
        assert len(padded) == max_length
        assert padded[:3] == [1, 2, 3]
        assert all(idx == dataset.pad_idx for idx in padded[3:])

    def test_image_caption_mappings(self, temp_data_dir):
        dataset = FlickrDataset(data_path=temp_data_dir)

        image_name = 'test_image1.jpg'
        caption_count = dataset.get_image_captions_count(image_name)
        all_captions = dataset.get_all_captions_for_image(image_name)

        assert caption_count == 3
        assert len(all_captions) == 3
        assert all(isinstance(caption, str) for caption in all_captions)

    def test_different_configurations(self, temp_data_dir):
        dataset1 = FlickrDataset(data_path=temp_data_dir, remove_stopwords=False)
        dataset2 = FlickrDataset(data_path=temp_data_dir, min_word_freq=1)
        dataset3 = FlickrDataset(data_path=temp_data_dir, is_train=False)

        assert len(dataset1) == len(dataset2) == len(dataset3)
        assert dataset2.get_vocab_size() >= dataset1.get_vocab_size()

    def test_collate_function(self, temp_data_dir):
        dataset = FlickrDataset(data_path=temp_data_dir)
        batch = [dataset[i] for i in range(3)]
        collated = collate_fn(batch)

        assert 'images' in collated
        assert 'captions' in collated
        assert 'caption_lengths' in collated
        assert 'image_names' in collated
        assert 'caption_texts' in collated

        assert collated['images'].shape == (3, 3, 224, 224)
        assert collated['captions'].shape[0] == 3
        assert len(collated['caption_lengths']) == 3
        assert len(collated['image_names']) == 3
        assert len(collated['caption_texts']) == 3

    def test_missing_image_handling(self, temp_data_dir):
        missing_image_path = os.path.join(temp_data_dir, 'images', 'test_image1.jpg')
        if os.path.exists(missing_image_path):
            os.remove(missing_image_path)

        dataset = FlickrDataset(data_path=temp_data_dir)

        sample = dataset[0]
        assert torch.is_tensor(sample['image'])
        assert sample['image'].shape == (3, 224, 224)

    def test_edge_cases(self, temp_data_dir):
        """Test edge cases and error conditions."""
        dataset = FlickrDataset(data_path=temp_data_dir)

        empty_caption = torch.tensor([dataset.pad_idx] * 10)
        decoded = dataset.decode_caption(empty_caption)
        assert decoded == ""
        special_caption = torch.tensor([dataset.start_idx, dataset.end_idx, dataset.pad_idx])
        decoded = dataset.decode_caption(special_caption)
        assert '<START>' in decoded and '<END>' in decoded

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, temp_data_dir, batch_size):
        """Test dataset with different batch sizes."""
        from torch.utils.data import DataLoader

        dataset = FlickrDataset(data_path=temp_data_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )

        batch = next(iter(dataloader))
        expected_batch_size = min(batch_size, len(dataset))
        assert batch['images'].shape[0] == expected_batch_size

    def test_transforms_applied(self, temp_data_dir):
        train_dataset = FlickrDataset(data_path=temp_data_dir, is_train=True)
        val_dataset = FlickrDataset(data_path=temp_data_dir, is_train=False)

        train_sample = train_dataset[0]
        val_sample = val_dataset[0]
        assert train_sample['image'].shape == val_sample['image'].shape
        # Get image size from config or default
        config = get_config()
        image_size = config.get('dataset.image_size', 224)
        assert train_sample['image'].shape == (3, image_size, image_size)

        assert train_sample['image'].max() <= 3.0  # Roughly normalized range
        assert train_sample['image'].min() >= -3.0
