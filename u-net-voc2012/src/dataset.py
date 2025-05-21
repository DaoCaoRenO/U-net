import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.masks = []
        self._load_data()

    def _load_data(self):
        image_dir = os.path.join(self.root_dir, 'JPEGImages')
        mask_dir = os.path.join(self.root_dir, 'SegmentationClass')
        
        for img_name in os.listdir(image_dir):
            if img_name.endswith('.jpg'):
                self.images.append(os.path.join(image_dir, img_name))
                mask_name = img_name.replace('.jpg', '.png')
                self.masks.append(os.path.join(mask_dir, mask_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])