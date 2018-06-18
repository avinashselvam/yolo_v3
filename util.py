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

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):

    # get diagonal coordinates of 2 bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # calculate the diagonal coords of intersection box
    # b3 refers to the intersection box formed
    b3_x1 = torch.max(b1_x1, b2_x1)
    b3_x2 = torch.max(b1_x2, b2_x2)
    b3_y1 = torch.min(b1_y1, b2_y1)
    b3_y2 = torch.min(b1_y2, b2_y2)

    # calc intersection area
    # clamp sides to a min = 0, case of no common area
    b3_area = torch.clamp(b3_x2 - b3_x1 + 1, min=0) * torch.clamp(b3_y2 - b3_y1 + 1, min=0)
    
    # calc union area
    

    
    

    return iou

def write_results(prediction, confidence, num_classes, nms_conf=0.4):

    # confidence thresholding
    confidence_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * confidence_mask
    
    # Non maximum supression needs 2 diagonal pts.
    # center +- width/2, center +- height/2 -> 2 diagonal pts
    # create temp variable to prevent read/write errors
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
    box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
    box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
    box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3]/2
    prediction[:,:,:4] = box_corner[:,:,:4]

    # different images have different no. of true detections
    # so iterate through each image as vectorisation isn't feasible
    batch_size = prediction.size(0)
    
    write = False
    
    # iterate through each prediction in the bbox table
    for ind in range(batch_size):
        # prediction of each image at network end
        image_pred = prediction[ind]

        # chuck minority classes' scores
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # extend axis 1 so that they can be appended
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        # elements to be concatenated
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # chuck rows that fail confidence thresholding
        non_zero_ind = torch.nonzero(image_pred[:, 4])
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1, 7)
        except:
            continue
        # if image_pred_ is empty
        if image_pred_.shape[0] == 0:
            continue

        # different classes detected in the image
        # refer above for defn. of unique TLDR: uses np.unique()
        img_classes = unique(image_pred_[:, -1])

        # perform NMS for each detected class
        for cls in img_classes:
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            
            
        
        