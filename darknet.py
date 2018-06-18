import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from util import *

def get_test_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (416, 416))
    img_ = img[:,:,::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis,:,:,:] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

def parse_cfg(path):

    # stores each line in a list, removing comments and empty lines
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]


    # read cfg and store each type of block as a dict
    block = {}
    blocks = []

    for line in lines:
        # sees a new type
        # append the values of block containing values of prev type
        # empty the block placeholder
        # create new type block 
        if line[0] == '[':  
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        # if not a type, add each attributes one by one to block
        else:
            key, value = line.split('=') 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block) # appends final block

    return blocks

def create_modules(blocks):

    net_info = blocks[0] # first element contains architecture info 
    module_list = nn.ModuleList() # used to store nn.module objects 
    prev_filters = 3 # keep track of depth of input
    output_filters = [] # keep track of all depths

    class DetectionLayer(nn.Module):
        def __init__(self, anchors):
            super(DetectionLayer, self).__init__()
            self.anchors = anchors
    
    class EmptyLayer(nn.Module):
        def __init__(self):
            super(EmptyLayer, self).__init__()

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        ### for convolutional block
        if x['type'] == 'convolutional':
            activation = x['activation']
            
            # since BatchNorm is optional
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True

            # get all attributes
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            # calculate padding
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)

            # check batch_norm and add
            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            # check activation and add
            if activation == 'leaky':
                an = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), an)
        
        ### for upsampling layer
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{0}'.format(index), upsample)

        ### for route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            
            # check if second layer is given
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            
            # if start/end index in +ve notation -> a[9]
            # bring it to -ve notation -> a[-1]
            if start > 0:
                start -= index
            if end > 0:
                end -= index
            
            # create empty layer
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)

            # concatenate / pass
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        ### for shortcut layer
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        elif x['type'] == 'yolo':

            # mask says which anchors to use
            mask = x['mask'].split(',')
            mask = [int(i) for i in mask]

            # masked ones selected from list of all anchors
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list

# prints summary of architecture            
# blocks = parse_cfg('./cfg/yolov3.cfg')
# print(create_modules(blocks))

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, CUDA):
        # first block is net info
        # rest all contain information of each layer of the network
        modules = self.blocks[1:]
        # used to cache outputs to be later used in route layers
        outputs = {}

        first_detection = 0

        for i, module in enumerate(modules):
            module_type = (module['type'])

            # conv2d and upsampling2d are predefined by torch
            # use them as sequential modules from module_list
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            
            # use outputs cache dict and provide appropiate values
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                # convert to -ve index notation a[9] -> a[-1]
                if layers[0] > 0:
                    layers[0] -= i
        
                # if only one index given get values of that layer
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                # if two are given concatenate values
                else:
                    # convert to -ve index notation a[9] -> a[-1]
                    if layers[1] > 0:
                        layers[1] -= i
                    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    # concatenate along depth axis
                    x = torch.cat((map1, map2), 1)
            
            # skip layer like in Resnet
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]
                
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                
                input_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = x.data
                x = predict_transform(x, input_dim, anchors, num_classes)
                # since we can't append to empty tensor
                if not first_detection:
                    detections = x
                    first_detection = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

model = Darknet('cfg/yolov3.cfg')
ip = get_test_input()
pred = model(inp)
print(pred)