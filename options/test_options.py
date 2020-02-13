from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # epochs
        self.parser.add_argument("--max_steps", type=int, help="number of training steps of discriminator(0 to disable)")
        self.parser.add_argument("--max_epochs", type=int, default=1, help="number of training epochs")
        self.parser.add_argument("--summary_freq", type=int, default=1, help="update summaries every summary_freq steps")
        self.parser.add_argument("--save_freq", type=int, default=500, help="save model every save_freq steps, 0 to disable")
        # continue?
        self.parser.add_argument('--load_checkpoint', type=str, default=None, help='continue training: path to the last checkpoint model to load')
        # scheduler set-up
        self.parser.add_argument('--imgs_before_doubling', default=8000, help='Number of real images to show before doubling the resolution.')
        self.parser.add_argument('--imgs_while_fading', default=4000, help='Number of real images to show when fading in new layers.')
        self.parser.add_argument('--batch_per_res', default={8:120, 16:60, 32:30, 64:15, 128:10, 256:5, 512:4, 1024:3}, help='Resolution-specific overrides.')
        # adam parameters
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.99, help='momentum term of adam')
        self.parser.add_argument('--epsilon', type=float, default=1e-8, help='momentum term of adam')
        # for discriminators        
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching')        
        self.parser.add_argument('--sparse_D', action='store_true', help='use sparse temporal discriminators to save memory')
        # for temporal
        self.parser.add_argument('--lambda_T', type=float, default=10.0, help='weight for temporal loss')
        self.parser.add_argument('--lambda_F', type=float, default=10.0, help='weight for flow loss')
        self.parser.add_argument('--n_frames_D', type=int, default=3, help='number of frames to feed into temporal discriminator')        
        self.parser.add_argument('--n_scales_temporal', type=int, default=2, help='number of temporal scales in the temporal discriminator')        
        self.parser.add_argument('--max_frames_per_gpu', type=int, default=1, help='max number of frames to load into one GPU at a time')
        self.parser.add_argument('--max_frames_backpropagate', type=int, default=1, help='max number of frames to backpropagate') 
        self.parser.add_argument('--max_t_step', type=int, default=1, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        self.parser.add_argument('--n_frames_total', type=int, default=30, help='the overall number of frames in a sequence to train with')                
        self.parser.add_argument('--niter_step', type=int, default=5, help='how many epochs do we change training batch size again')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='if specified, only train the finest spatial layer for the given iterations')
        # misc
        self.parser.add_argument('--pool_size', type=int, default=1, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs at a time')

        self.isTrain = True
