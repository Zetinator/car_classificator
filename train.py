import time
import os
import sys
import copy
from pathlib import Path

import torch
import torchvision

import numpy as np
from matplotlib import pyplot as plt

from options.train_options import TrainOptions
from models.data_loader import CarLoader
from models.network import Classifier

def imshow(img: torch.Tensor):
    """show a grid of images
    """
    # un-normalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main(argv):
    # ==============================
    # SET-UP
    # ==============================
    # load dataset
    data_loader = CarLoader(opt)
    dataset = data_loader.load_train()
    # load network
    model = Classifier(opt)
    # load loss function
    criterion = torch.nn.CrossEntropyLoss()
    # move to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda(); criterion.cuda();
    # load optimizer
    optimizer = torch.optim.Adam(model.model.parameters(),
                                 lr=opt.lr,
                                 betas=(opt.beta1, opt.beta2))
    # load learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=opt.lr_step_size,
                                                gamma=opt.lr_update_rate)
    # ==============================
    # TRAINING
    # ==============================
    for epoch in range(opt.max_epochs):
        # epoch set-up
        since = time.time()
        total = 0
        current_loss = 0.0
        accuracy = 0.0
        # iterate through dataset
        for i, (imgs, labels) in enumerate(iter(dataset)):
            # move to avalible gpu
            imgs = imgs.to(device); labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            predictions = model(imgs)
            loss = criterion(predictions, labels)
            # update weights through backpropagation
            loss.backward()
            optimizer.step()
            # update learning rate
            scheduler.step()
            # keep records
            total += labels.size(0)
            current_loss += loss.item()
            # import pdb; pdb.set_trace()
            _, predictions = torch.max(predictions.data, 1)
            accuracy += (predictions == labels).sum().item()
            print(f'\rprocessing: {i}/{len(dataset)}, loss: {loss.item(): >10}', end='')
        # save checkpoint, create dir if not already there
        if epoch % opt.save_freq == 0 or epoch == opt.max_epochs-1:
            Path(opt.checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.model.state_dict(), os.path.join(opt.checkpoint, f'e{epoch}_l{loss}'))
        # export summary
        current_loss = current_loss/total
        accuracy = accuracy/total
        # print summary
        print('\repoch: {e:>6}, loss: {loss:.4f}, accuracy: {acc:.4f}, in: {time:.4f}s'\
                .format(e=epoch,
                        loss = current_loss,
                        acc = accuracy,
                        time = time.time()-since))

if __name__ == '__main__':
    # set-up, parse options
    opt = TrainOptions().parse()
    main(sys.argv)
