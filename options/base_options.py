"""general options to pass when initializing
all the options you would pass to the command line are here
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # general
        self.parser.add_argument("--train_dataset", default='./data/data_in_class_folder', help="path to the dataset of images to train")
        self.parser.add_argument("--test_dataset", default='./data/data_in_class_folder', help="path to the dataset of images to train")
        self.parser.add_argument("--logs", default='./logs', help="path to logs dir, to print in tensorboard")
        self.parser.add_argument("--output_dir", default='./results', help="where to put output files")
        self.parser.add_argument("--checkpoint", default='./checkpoints', help="directory with checkpoint to resume training from or use for testing")
        # data
        self.parser.add_argument("--batch_size", type=int, default=32, help="number of images in batch")
        self.parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
        self.parser.add_argument("--num_classes", type=int, default=196,  help="number of classes on the dataset to classify")
        # misc
        self.parser.add_argument("--num_channels", type=int, default=3, help="Number of output color channels")
        self.parser.add_argument("--dtype", default='float32', help="Data type to use for activations and outputs.")

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
