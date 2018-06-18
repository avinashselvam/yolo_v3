import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():

    parser = argparse.ArgumentParser(description='YOLO v3 Inference module')

    parser.add_argument('--images', dest='images', help='directory of images to be inferred', type=str)
    parser.add_argument('--det', dest='det', help='directory to write inferences to', type=str)
    parser.add_argument('--bs', dest='bs', help='Batch size', default=1)
    parser.add_argument('--confidence', dest='confidence', help='threshold for object confidence score', default=0.5)
    parser.add_argument('--nms', dest='nms_thresh', help='threshold for Non maximum suppression', default=0.4)
    parser.add_argument('--cfg', dest='cfgfile', help='path to config file', type=str)
    parser.add_argument('--weights', dest='weightsfile', help='path to weights file', type=str)
    parser.add_argument('--res', dest='res', help='Input resolution of the network', default='416', type=str)

    return parser.parse_args()

args=arg_parse()

images=args.images
batch_size=int(args.bs)
confidence=float(args.confidence)
nms_thresh=float(args.nms_thresh)
start=0
