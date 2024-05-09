import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = "images"
        self.mask_folder = "masks"
        self.classes = 5  # Number of classes
        
        self.image_path_list = os.listdir(os.path.join(self.root_dir, self.image_folder))
        self.mask_path_list = os.listdir(os.path.join(self.root_dir, self.mask_folder))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_name = self.image_path_list[idx]
        mask_name = self.mask_path_list[idx]
        
        img_path = os.path.join(self.root_dir, self.image_folder, img_name)
        mask_path = os.path.join(self.root_dir, self.mask_folder, mask_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Convert mask to binary tensor with one-hot encoding
        mask = self.convert_to_one_hot(mask)
        
        return image, mask

    def convert_to_one_hot(self, mask):
        mask_one_hot = torch.zeros((self.classes, *mask.size))
        for c in range(self.classes):
            mask_one_hot[c][mask == c] = 1
        return mask_one_hot
