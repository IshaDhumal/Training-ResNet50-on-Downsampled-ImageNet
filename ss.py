import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from PIL import ImageFile
from ptflops import get_model_complexity_info

from datasets import smallimagenet, tinyimagenet
from models.resnets import ResidualNet

def save_checkpoint(state, is_best, prefix):
    if not os.path.exists('/home/isha/CheckpointFolder/checkpoints'):
        os.mkdir('/home/isha/CheckpointFolder/checkpoints')
    filename = '/home/isha/CheckpointFolder/checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/home/isha/CheckpointFolder/checkpoints/%s_model_best.pth.tar' % prefix)
        wandb.save(filename)

save_checkpoint({
            'epoch':1,
            'state_dict': 'one',
            'best_prec1': 55,
            'optimizer': 'cool',
        }, '0', 'test')
        
        
