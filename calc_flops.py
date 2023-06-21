from thop import profile
import os
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from conf import settings
from utils import get_network, get_test_dataloader

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
args = parser.parse_args()
net = get_network(args)
net.load_state_dict(torch.load(args.weights))
net.eval()

image = torch.randn(1, 3, 1024, 1392).cuda()
macs, params = profile(net, inputs=(image, ))
print('*'*30 + '{}'.format(args.net) + '*'*30)
print('macs:\n{}\n'.format(macs))
print('params:\n{}\n'.format(params))
