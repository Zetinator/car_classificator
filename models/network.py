"""defines the network architecture
given the limited number of images in the train set a pretrained network
will be used for the feature extraction...

the base model in considareation is the "mobilenet_v2"
https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
"""
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

from models.classes import classes 

class Classifier(torch.nn.Module):
    """classificator of cars
    """
    def __init__(self, opt=None):
        super(Classifier, self).__init__()
        self.opt = opt
        self.model = None
        self.classes = classes
        self.build()

    def build(self):
        """build our custom network over the mobilenet_v2
        """
        # load pretrained mobilenet_v2
        model = models.mobilenet_v2(pretrained=True)
        # allow fine-tuning
        # for param in model.parameters():
            # param.requires_grad = False
        # the last module of the mobilenet_v2 is called classifier
            # (0): Dropout(p=0.2, inplace=False)
            # (1): Linear(in_features=1280, out_features=1000, bias=True)
        in_features = model.classifier[1].in_features
        out_features = self.opt.num_classes if self.opt else len(self.classes)
        # we add our custom fully connected layer according to the num of classes
        model.classifier[1] = torch.nn.Linear(in_features, out_features)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def predict(self, img, show_probability=False):
        """same as forward but with no backpropagation
        """
        input_image = Image.open(img)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)
        # deactivate regularizers and backpropagation
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_batch)
        _, preds = torch.max(prediction, 1)
        preds = preds.item()
        if not show_probability:
            return f'{self.classes[preds]}'
        p = {k:v.item() for k,v in zip(self.classes, F.softmax(prediction[0], dim=0))}
        return f'{self.classes[preds]} with:\n{p}'
