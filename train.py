import time
import os
import sys
import copy
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt

from options.train_options import TrainOptions
from models.data_loader import CarLoader
from models.network import Classifier

def probabilities(predictions, labels):
    """convert output probabilities to predicted class
    """
    _, preds = torch.max(predictions, 1)
    preds = np.squeeze(preds.cpu().numpy())
    return preds, [F.softmax(chosen_one, dim=0)[i].item()
                        for i, chosen_one in zip(preds, predictions)]
def _plot_grid(predictions, imgs, labels, classes):
    """Generates matplotlib Figure
    that shows the network's top predictions along with its probability
    """
    preds, probs = probabilities(predictions, imgs)
    # plot the imgs in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(24, 12))
    imgs = imgs/2 + 0.5
    np_imgs = imgs.cpu().numpy()
    np_imgs = np.transpose(np_imgs, (0, 2, 3, 1))
    n = labels.size(0)
    for i in range(n):
        ax = fig.add_subplot(n//4, 4, i+1, xticks=[], yticks=[])
        plt.imshow(np_imgs[i].clip(0,1))
        ax.set_title('{0}, {1:.2}%\nground truth: {2}'.format(
            classes[preds[i]],
            probs[i] * 100.0,
            classes[labels[i]]),
            color=("green" if preds[i]==labels[i].item() else "red"))
    return fig

def main(argv):
    # ==============================
    # SET-UP
    # ==============================
    # create summary writer, publisher
    Path(opt.logs).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(opt.logs,f'lr{opt.lr}_{time.time()}'))
    # load dataset
    data_loader = CarLoader(opt)
    dataset = data_loader.load_train()
    # load network, and publish
    model = Classifier(opt)
    writer.add_graph(model.model, torch.randn(1,3,224,224))
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
            _, preds = torch.max(predictions.data, 1)
            accuracy += (preds == labels).sum().item()
            print(f'\rprocessing: {i}/{len(dataset)}, loss: {loss.item(): >10}', end='')
        # save checkpoint, create dir if not already there
        if epoch % opt.save_freq == 0 or epoch == opt.max_epochs-1:
            Path(opt.checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.model.state_dict(), os.path.join(opt.checkpoint, f'e{epoch}_l{loss}'))
        # export summary
        current_loss = current_loss/total
        accuracy = accuracy/total
        # publish to tensorflow
        if epoch % opt.summary_freq == 0 or epoch == opt.max_epochs-1:
            writer.add_scalar('loss', current_loss, epoch)
            writer.add_scalar('accuracy', accuracy, epoch)
            # publish grid with predictions
            fig = _plot_grid(predictions, imgs, labels, data_loader.classes)
            writer.add_figure('predictions', fig, global_step=epoch)
            for tag, parm in model.model.named_parameters:
                writer.add_histogram(f'grad_' + tag, parm.grad.data.cpu().numpy(), epoch)
                writer.add_histogram(f'weights_' + tag, parm.weights.data.cpu().numpy(), epoch)
            # writer.add_histogram('classifier', model.model.classifier[1].weight, global_step=epoch)
        # print summary
        print('\repoch: {e:>6}, loss: {loss:.4f}, accuracy: {acc:.4f}, in: {time:.4f}s'\
                .format(e=epoch,
                        loss = current_loss,
                        acc = accuracy,
                        time = time.time()-since))
    writer.close()

if __name__ == '__main__':
    # set-up, parse options
    opt = TrainOptions().parse()
    main(sys.argv)
