from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys

import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt

from options.train_options import TrainOptions
from models.data_loader import CarLoader
from models.network import *

# ------------------------------------------------------------
# define training step as function
# ------------------------------------------------------------
def train_step():
    return

def imshow(img: torch.Tensor):
    """show a grid of images
    """
    # un-normalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train_step(model, criterion, optimizer, scheduler):
    since = time.time()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            checkpoint = copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(checkpoint)
    return model

def main(argv):
    # set-up
    print('loading dataset...')
    data_loader = CarLoader(opt)
    dataset = data_loader.load()
    dataiter = iter(dataset)
    images, labels = dataiter.next()

    checkpoint = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(opt.max_epochs):
        print(f'epoch {epoch}/{opt.max_epochs}')
        train_step()

    print(f'GroundTruth:')
    print(f'{[data_loader.classes[e.item()] for e in labels]}')
    # print images
    imshow(torchvision.utils.make_grid(images))


if __name__ == '__main__':
    # set-up, parse options
    opt = TrainOptions().parse()
    main(sys.argv)
