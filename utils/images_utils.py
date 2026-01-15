import os
from PIL import Image
from torchvision.transforms import v2

def get_image_by_name(data_path, images_path, name, transform=None):
    image_path = os.path.join(data_path, images_path, name)
    image_file = Image.open(image_path).convert('RGB')
    if transform:
        image_file = transform(image_file)

    return image_file

def make_transform(is_train):
    if is_train:
        transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(mean=[0.4463, 0.4506, 0.4166], std=[0.2235, 0.2138, 0.2222]),
        ])
    else:
        transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.4463, 0.4506, 0.4166], std=[0.2235, 0.2138, 0.2222]),
        ])
    return transform
