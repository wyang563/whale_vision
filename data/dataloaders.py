import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
from configs import load_config

class SegmentorDataset(Dataset):
    def __init__(self, config):
        self.data_path = config["data_path"]
        self.image_files = os.listdir(self.data_path)
        self.input_shape = config["model"]["input_shape"]
        running_mode = config["run"]["mode"]

        # set transforms
        data_transforms = []
        data_transforms.append(transforms.Resize(self.input_shape))
        if running_mode == "train":
            data_transforms.append(transforms.RandomHorizontalFlip(p = 0.5 if config["augmentation"]["horizontal_flip"] else 0))
            data_transforms.append(transforms.RandomVerticalFlip(p = 0.5 if config["augmentation"]["vertical_flip"] else 0))        
        
        if config["augmentation"]["normalize"]:
            data_transforms.append(transforms.ToTensor())

        data_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transforms = transforms.Compose(data_transforms)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # split image into input and target
        if image.size != (1000, 300):
            image = image.resize((1000, 300))
        input_image = image.crop((0, 0, 500, 300))
        target_image = image.crop((500, 0, 1000, 300))
        input_image = self.transforms(input_image)
        target_image = self.transforms(target_image)
        return input_image, target_image
    
class RecognitionDataset(Dataset):
    def __init__(self, config):
        self.data_path = config["data_path"]
        self.positive_image_files = os.listdir(os.path.join(self.data_path, "positive"))
        self.negative_image_files = os.listdir(os.path.join(self.data_path, "negative"))

        self.input_shape = config["model"]["input_shape"]
        running_mode = config["run"]["mode"]
        data_transforms = []
        data_transforms.append(transforms.Resize(self.input_shape))
        if running_mode == "train":
            data_transforms.append(transforms.RandomHorizontalFlip(p = 0.5 if config["augmentation"]["horizontal_flip"] else 0))
            data_transforms.append(transforms.RandomVerticalFlip(p = 0.5 if config["augmentation"]["vertical_flip"] else 0))        
        
        if config["augmentation"]["normalize"]:
            data_transforms.append(transforms.ToTensor())

        data_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transforms = transforms.Compose(data_transforms)

    def __len__(self):
        return len(self.positive_image_files) + len(self.negative_image_files)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_image_files):
            img_path = os.path.join(self.data_path, "positive", self.positive_image_files[idx])
            label = 1
        else:
            img_path = os.path.join(self.data_path, "negative", self.negative_image_files[idx - len(self.positive_image_files)])
            label = 0 
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        return image, label

def get_dataloader(config):
    if config["run"]["task"] == "segmentation":
        dataset = SegmentorDataset(config)
    elif config["run"]["task"] == "recognition":
        dataset = RecognitionDataset(config)
    else:
        raise Exception("Invalid train run task mode")

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
    )

if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    dataloader = get_dataloader(config)