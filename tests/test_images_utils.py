import os
import pytest
import torch
import tempfile
from torchvision.transforms import v2
from utils.images_utils import make_transform, get_image_by_name
from PIL import Image

class TestMakeTransform:
    def test_make_transform_train_returns_compose(self):
        transform = make_transform(is_train=True)
        assert isinstance(transform, v2.Compose)
    
    def test_make_transform_val_returns_compose(self):
        transform = make_transform(is_train=False)
        assert isinstance(transform, v2.Compose)
    
    def test_make_transform_resizes_image(self):
        test_image = Image.new('RGB', (100, 100), color='red')
        transform = make_transform(is_train=True)
        transformed = transform(test_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 224, 224)


class TestGetImageByName:
    @pytest.fixture
    def temp_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = os.path.join(temp_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            test_image = Image.new('RGB', (100, 100), color='blue')
            image_path = os.path.join(images_dir, 'test.jpg')
            test_image.save(image_path)
            
            yield temp_dir, 'images', 'test.jpg' 
    
    def test_is_image(self, temp_dataset):
        data_path, images_path, name = temp_dataset
        image_from_name = get_image_by_name(data_path, images_path, name)
        
        assert isinstance(image_from_name, Image.Image)
    
    def test_is_rgb(self, temp_dataset):
        data_path, images_path, name = temp_dataset
        image_from_name = get_image_by_name(data_path, images_path, name)
        
        assert image_from_name.mode == 'RGB'
    
    def test_with_transform(self, temp_dataset):
        data_path, images_path, name = temp_dataset
        transform = make_transform(is_train=True)
        image_from_name = get_image_by_name(data_path, images_path, name, transform=transform)
        
        assert isinstance(image_from_name, torch.Tensor)
        
        
        