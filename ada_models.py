from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import create_modules, Upsample, EmptyLayer, YOLOLayer

class AdaptiveYOLO(nn.Module):
    """Adaptive YOLO object detection model"""

    def __init__(self, config_path, img_size=416):
        super(AdaptiveYOLO, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        self.mode = 0
        self.mode_dicts_class_to_cluster = None
        self.mode_classes_list = None
        self.num_all_classes = 0

    def forward(self, x, targets=None):
        # Modify the targets to the cluster targets
        modified_targets = None
        if targets != None:
            # Convert the original labels to internal cluster based labels

            modified_targets = targets.clone()
            active_classes = self.mode_classes_list[self.mode]
            # print("Active classes are ", active_classes)
            print("Class to Cluster mapping is ", self.mode_dicts_class_to_cluster[self.mode])
            to_delete = []
            for i, ele in enumerate(modified_targets[:,1]):
                if ele in active_classes:
                    modified_targets[i,1] = torch.tensor(self.mode_dicts_class_to_cluster[self.mode][int(ele)], dtype=torch.float64, device=x.get_device())
                else:
                    to_delete.append(i)
            # print(">>>>>>>>>>>> ", len(to_delete), " Deleted ", modified_targets.shape[0], "Remaining ")
            to_delete.sort(reverse=True)
            for i in to_delete:      
                if i == modified_targets.shape[0] -1: # Last element
                    modified_targets = modified_targets[0:i] # remove the last row
                elif i == 0: # First element
                    modified_targets = modified_targets[1:] # remove the first row
                else: # otherwise
                    modified_targets = torch.cat([modified_targets[0:i], modified_targets[i+1:]]) # remove the current row

            if modified_targets.shape[0] == 0:
                return (0, None)

        # print("Modified Targets are ", modified_targets[:,1], modified_targets.shape)

        img_dim = x.shape[2]
        loss = 0
        split = False
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # print(i, module_def["type"])
            if split == True and module_def["type"] != "split2":
                continue
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "split1":
                if self.mode == 0:
                    x = module(x)
                else:
                    split = True
                    continue
            elif module_def["type"] == "split2":
                if self.mode == 0:
                    break
                else:
                    x = module(x)
                    split = False
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, modified_targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)


        # Convert back the internal cluster based labels to original labels
        for l, layer in enumerate(yolo_outputs):
            temp = torch.zeros(layer.shape[0],layer.shape[1],self.num_all_classes+5-layer.shape[-1], device=x.get_device())
            yolo_outputs[l] = torch.cat((layer,temp), dim=2)

            full_detection = torch.zeros(layer.shape[0],layer.shape[1], self.num_all_classes+5, device=x.get_device())   
            full_detection[:, :, 0:5] = layer[:, :, 0:5]
            new_indices = [5 + k for k in self.mode_classes_list[self.mode]]
            full_detection[:, :, new_indices] = layer[:, :, 5:]
            yolo_outputs[l] = full_detection

        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path, layer_cutoff_idx=0):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = 0
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        cutoff = min(cutoff, layer_cutoff_idx)
        
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
