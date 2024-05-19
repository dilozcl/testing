import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_transform=None, class_encoding=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform
        self.class_encoding = class_encoding

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = np.array(mask)
            mask = self.encode_segmap(mask)

        return image, mask

    def encode_segmap(self, mask):
        mask_encoded = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for cls, color in self.class_encoding.items():
            mask_encoded[(mask == color).all(axis=2)] = cls
        return mask_encoded

# Example usage:
# class_encoding = {0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0)}  # Map RGB to class index
# dataset = SegmentationDataset(image_paths, mask_paths, transform=transforms.ToTensor(), class_encoding=class_encoding)
