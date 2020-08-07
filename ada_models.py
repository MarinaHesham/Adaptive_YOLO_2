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

from models import Upsample, EmptyLayer, YOLOLayer

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    split_idx = -1
    new_branch = False
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = int(module_def["pad"])
            if new_branch:
                in_filters = output_filters[split_idx]
                new_branch = False
            else:
                in_filters = output_filters[-1]
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=in_filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "branch":
            if split_idx == -1:
                split_idx = module_i-1
            modules.add_module(f"branch_{module_i}", EmptyLayer())
            new_branch = True

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        else:
            print("ERRRRRRRROR ", module_def["type"])
        # Register module list and number of output filters
        module_list.append(modules)
        print(module_list[-1])
        output_filters.append(filters)

    return hyperparams, module_list

class ClusterModel(nn.Module):
    def __init__(self,config_path, out_classes, img_size=416):
        super(ClusterModel, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.num_all_classes = out_classes
        self.seen = 0
        self.fc1 = nn.Linear(144, 64)
        self.fc2 = nn.Linear(64, self.num_all_classes)

    def forward(self, x):
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            layer_outputs.append(x)
            #print(i, module_def["type"], x.shape)
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        x = F.leaky_relu(self.fc1(x),0.1)
        x = self.fc2(x)
        return F.softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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
        skip = False
        layer_outputs, yolo_outputs = [], []
        current_branch = -1
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if skip == True and module_def["type"] != "branch":
                continue
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "branch":
                current_branch += 1
                if self.mode != current_branch:
                    skip = True
                    continue
                else:
                    skip = False
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
            #print(i, module_def["type"], x.shape)


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
