import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):

    # e.g prediction.size = [1000, 7, 7, B*(5+C)]
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # unsure why can't do this in one step, maybe depends on how view works
    # [1000, 7, 7, B*(5+C)] -> [1000, B*(5+C), 7*7]
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    # [1000, B*(5+C), 7*7] -> [1000, 7*7, B*(5+C)]
    prediction = prediction.transpose(1, 2).contiguous()
    # [1000, 7*7, B*(5+C)] -> [1000, 7*7*B, (5+C)]
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # convert anchor dimensions from input scale to output scale
    # i.e divide it by stride
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    
    # apply sigmoid to center (x, y), objectness score & class scores
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    prediction[:,:,5: 5+num_classes] = torch.sigmoid(prediction[:,:,5: 5+num_classes])
    
    
    # apply center offsets
    # this part is absolutely brilliant
    grid = np.arange(grid_size)
    # a[0] = [0, 1, 2 ...]; b[0] = [0, 0, 0 ...]
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1) # [49, 1]
    y_offset = torch.FloatTensor(b).view(-1, 1) # [49, 1]

    # [49, 2] -> [49, 2*B] -> [49*B, 2] -> [1, 49*B, 2]
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # apply log transform to height & width
    anchors = torch.FloatTensor(anchors)  # [B, 2]
    # [49*B, 2] -> [1, 49*B, 2]
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:, 2:4] = torch.exp(prediction[:,:, 2:4]) * anchors
    
    # resize dimensional values to input scale
    prediction[:,:,:4] *= stride

    return prediction

def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    