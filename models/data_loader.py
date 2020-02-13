"""defines hoe to preprocess and load the images from the dataset
"""
import json

import torch
from torchvision import transforms, datasets

class CarLoader():
    """standard preprocessing for the mobilenet_v2
    https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    """
    def __init__(self, opt):
        self.opt = opt
        self.classes = None
        self.class_to_idx = None

    def load(self):
        """call this function to load the dataset
        """
        # define chain of preprocessing steps
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomPerspective(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # apply preprocessing
        data = datasets.ImageFolder(root=self.opt.dataset,
                                             transform=preprocess)
        self.classes = data.classes
        self.class_to_idx = data.class_to_idx
        # return DataLoader initialized
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.opt.batch_size,
                                           shuffle=True,
                                           num_workers=self.opt.num_workers)
