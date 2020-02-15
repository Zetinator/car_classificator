from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # epochs
        self.parser.add_argument("--max_epochs", type=int, default=205, help="number of training epochs")
        self.parser.add_argument("--summary_freq", type=int, default=5, help="update summaries every summary_freq steps")
        self.parser.add_argument("--save_freq", type=int, default=10, help="save model every save_freq epochs, 0 to disable")
        # continue?
        self.parser.add_argument('--load_checkpoint', type=str, default=None, help='continue training: path to the last checkpoint model to load')
        # adam parameters
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--lr_step_size', type=float, default=20, help='initial learning rate for adam')
        self.parser.add_argument('--lr_update_rate', type=float, default=0.1, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.99, help='momentum term of adam')
        self.parser.add_argument('--epsilon', type=float, default=1e-8, help='momentum term of adam')
        # for discriminators        
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching')        
        self.parser.add_argument('--sparse_D', action='store_true', help='use sparse temporal discriminators to save memory')
        # misc
        self.parser.add_argument('--num_gpu', type=int, default=0, help='number of GPUs at a time')

        self.isTrain = True
