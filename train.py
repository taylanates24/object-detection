import cv2
from models.model import TyNet
from check_anchors import check_anchors
from data.dataset import create_dataloader
imgsz = 320
import torch
import json
from torch.cuda import amp
from loss import ComputeLoss
f = open('hyp.json')
import numpy as np
import torchvision
from nms import non_max_suppression

from postprocessing import training_postprocess, validation_postprocess

hyp = json.load(f)
anchors = torch.tensor([[
         [ 10.,  13.],
         [ 16.,  30.],
         [ 33.,  23.]],

        [[ 30.,  61.],
         [62., 45.],
         [59., 119.]],
        [[ 116.,  90.],
         [156., 198.],
         [373., 326.]]
        ], device='cuda:0')


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']




