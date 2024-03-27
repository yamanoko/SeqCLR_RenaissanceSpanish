import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import RandomApply, GaussianBlur, Resize, Compose, ToTensor, Lambda, RandomPerspective, \
    RandomAffine
from PIL import Image
import os
import numpy as np
import random
from torch.utils.data import Dataset


class RandomVerticalCrop:
    def __init__(self, crop_height):
        self.crop_height = crop_height

    def __call__(self, img):
        (width, height) = img.size
        start = random.randint(0, height - self.crop_height)
        end = start + self.crop_height
        return img.crop((0, start, width, end))


class ContrastiveLearningDataset(Dataset):
    def __init__(self, image_dir, max_size=5000, crop_height=50, img_size=(50, 700)):
        super().__init__()
        self.max_size = max_size
        self.img_size = img_size
        assert os.path.isdir(image_dir)
        self.filepaths = [
            os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, filename))
        ]
        self.image_dir = image_dir
        self.original_transform = Compose([
            Lambda(lambda img: img.convert("RGB")),
            Resize(self.img_size),
            ToTensor(),
        ])
        self.augmented_transform = Compose([
            Lambda(lambda img: img.convert("RGB")),
            RandomApply([RandomVerticalCrop(crop_height=crop_height)], p=0.5),
            RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            RandomApply([RandomPerspective(distortion_scale=0.1)], p=0.5),
            RandomApply([RandomAffine(degrees=5)], p=0.5),
            Resize(self.img_size),
            ToTensor(),
        ])

    def __len__(self):
        return min(len(self.filepaths), self.max_size)

    def __getitem__(self, idx):
        try:
            with Image.open(self.filepaths[idx]) as img:
                img = img.convert("RGB")  # Added this line to convert images to RGB format
                img = img.resize(self.img_size)
        except IOError:
            return "cannot identify image file '%s'", self.filepaths[idx]
        original = self.original_transform(img)
        augmented = self.augmented_transform(img)
        return {"original": original, "augmented": augmented}


class DecoderDataset(Dataset):
    def __init__(self, csv_file, img_dir, token_dict, max_size=10000, max_length=50, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file, index_col=0)
        self.token_dict = token_dict
        self.max_size = max_size
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),  # Added this to convert images to RGB format
            transforms.Resize((50, 700)),


            transforms.ToTensor(),  # Convert image to PyTorch Tensor in CHW format
            *([transform] if transform else [])
        ])

    def __len__(self):
        return min(len(self.annotations), self.max_size)

    def __getitem__(self, index):
        img_name = self.annotations.iloc[-(index+1), 1]
        image = Image.open(os.path.join(self.img_dir, img_name))  # Use PIL to read the image
        image = self.transform(image)  # Image is in CHW format now

        label = self.annotations.iloc[-(index+1), 0]
        label_tokenized = [
            self.token_dict[char.lower()] if char.lower() in self.token_dict
            else self.token_dict["<UNK>"] for char in label]
        label_tokenized = label_tokenized[:self.max_length]
        label_tokenized.append(self.token_dict['<EOS>'])
        label_length = len(label_tokenized)
        for i in range(label_length, self.max_length+1):
            label_tokenized.append(self.token_dict['<PAD>'])
        return image, torch.tensor(label_tokenized, dtype=torch.long)
