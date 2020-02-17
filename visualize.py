import os
import sys
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from options.train_options import TrainOptions
from models.classes import classes

def main(argv):
    # ==============================
    # SET-UP
    # ==============================
    # create summary writer, publisher
    Path(opt.logs).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(os.path.join(opt.logs,f'data_visualization_embeddings'))
    # define chain of preprocessing steps
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # apply preprocessing
    data = datasets.ImageFolder(root=opt.train_dataset,
                                transform=preprocess)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=1)
    features, labels, imgs = None, [], []
    # ==============================
    # PUBLISH
    # ==============================
    # load 1000 random samples
    for i, (img, label) in enumerate(data_loader):
        if i == 100: break
        labels.append(classes[label])
        # flat the tensor
        feature = img.view(-1,3*224*224)
        if features is None:
            features = np.array(feature); continue
        features = np.append(features, feature, axis=0)
    # time to publish
    writer.add_embedding(features,
                         metadata=labels)
    writer.close()

if __name__ == '__main__':
    # set-up, parse options
    opt = TrainOptions().parse()
    main(sys.argv)
