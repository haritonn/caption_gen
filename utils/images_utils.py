import os
from PIL import Image
import torch
from torchvision.transforms import v2
from config_loader import get_config

def get_image_by_name(data_path, images_path, name, transform=None):
    image_path = os.path.join(data_path, images_path, name)
    image_file = Image.open(image_path).convert('RGB')
    if transform:
        image_file = transform(image_file)

    return image_file

def make_transform(is_train, config_path=None):
    config = get_config(config_path)
    dataset_config = config.get_dataset_config()

    image_size = dataset_config.get('image_size', 224)
    normalize_mean = dataset_config.get('normalize_mean', [0.4463, 0.4506, 0.4166])
    normalize_std = dataset_config.get('normalize_std', [0.2235, 0.2138, 0.2222])

    if is_train:
        augmentation = dataset_config.get('augmentation', {})
        flip_prob = augmentation.get('random_horizontal_flip', 0.5)

        transform = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=flip_prob),
            v2.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    else:
        transform = v2.Compose([
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    return transform
