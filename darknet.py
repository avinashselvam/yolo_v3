import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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

            # says which anchors to use
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
        modules = self.blocks[1:]
        outputs = {}