from __future__ import division

from ada_models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from models import Darknet
from ada_test import evaluate_branch
from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import random
from ptflops import get_model_complexity_info


def evaluate_branches(backbone, branches, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, max_bound=False, c_model=None, clusters_path = None):
    backbone.eval()
    for branch in branches:
        branch.eval()
    
    common_classes = [0,1,2,7,13,16,24,25,26,27,39,41,45,56,58,60,67,73,74]
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    clusters = parse_clusters_config(clusters_path)
    cluster_idx = 0
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()

        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        
        # Select mode
        if max_bound:
            # print(targets[:, 1].tolist())
            ts = targets[:, 1].tolist()
            cluster_cnt = np.zeros(len(clusters))
            for t in ts:
                for i, cluster in enumerate(clusters):
                    if t in cluster and t not in common_classes:
                        cluster_cnt[i] += 1
            # print("Cluster Count", cluster_cnt, "Chosen Max", np.argmax(cluster_cnt))
            # dominent_clus = argNmaxelements(cluster_cnt, 1)[0]
            # print(dominent_clus)
            # print("Neglected/All Objects = ", neglected_objects, "/", all_objects)
            # print(cluster_cnt)
            cluster_idx = np.argmax(cluster_cnt)
            # print(cluster_idx)
        else:
            cluster_idx = random.randint(0, 3)
        with torch.no_grad():            
            backbone_out = backbone(imgs)
            print(imgs.shape)
            outputs = branches[cluster_idx](backbone_out, img_size)
            temp = torch.zeros(len(outputs), outputs[0].shape[0], 80+5-outputs[0].shape[-1])
            outputs = torch.cat((outputs,temp), dim=2)
            new_indices = [5 + k for k in clusters[cluster_idx]]
            outputs[0][ :, new_indices] = outputs[0][ :, 5:5+len(new_indices)]
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--backbone_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--backbone_weights", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--backbone_num_of_layers", type=str, help="number of layers in backbone model, must be specified")
    parser.add_argument("--branch0_weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--branch1_weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--branch2_weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--branch3_weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")

    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--clusters_path", type=str, default="clusters.data", help="clusters file path")
    parser.add_argument("--hier_class", type=bool, default=False, help="when True enable hierarical classification")
    parser.add_argument("--hier_model_cfg", type=str, help="path to hierarical model congiguration")
    parser.add_argument("--hier_model", type=str, help="path to hierarical classification model")
    parser.add_argument("--hier_model_shared_layers", type=int, default=-1, help="path to hierarical classification model")
    parser.add_argument("--max_bound", type=bool, default=False, help="when True enable max bound for the hierarical model")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
 
    # Define Backbone Model and load its weights
    backbone = Backbone(opt.backbone_def).to(device)
    count_parameters(backbone)
    num_classes = int(data_config["classes"])
    if opt.backbone_weights:
        backbone.load_darknet_weights(opt.backbone_weights, int(opt.backbone_num_of_layers))

    print("BACKBONE ...... ")
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(backbone, (3, 416, 416), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    branches = []

    # Define Branch 0 and load its weights
    branch0 = Darknet("config/adayolo_branch32.cfg").to(device)
    count_parameters(branch0)

    if opt.branch0_weights_path:
        if opt.branch0_weights_path.endswith(".pth"):
            branch0.load_state_dict(torch.load(opt.branch0_weights_path))
            branches.append(branch0)

    # print("BRANCH 0 ...... ")
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(branch0, (1024, 13, 13), as_strings=True, print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Define Branch 1 and load its weights
    branch1 = Darknet("config/adayolo_branch32.cfg").to(device)
    count_parameters(branch1)

    if opt.branch1_weights_path:
        if opt.branch1_weights_path.endswith(".pth"):
            branch1.load_state_dict(torch.load(opt.branch1_weights_path))
            branches.append(branch1)

    # print("BRANCH 1 ...... ")
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(branch1, (1024, 13, 13), as_strings=True, print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Define Branch 2 and load its weights
    branch2 = Darknet("config/adayolo_branch34.cfg").to(device)
    count_parameters(branch2)

    if opt.branch2_weights_path:
        if opt.branch2_weights_path.endswith(".pth"):
            branch2.load_state_dict(torch.load(opt.branch2_weights_path))
            branches.append(branch2)

    # print("BRANCH 2 ...... ")
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(branch2, (1024, 13, 13), as_strings=True, print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Define Branch 3 and load its weights
    branch3 = Darknet("config/adayolo_branch39.cfg").to(device)
    count_parameters(branch3)

    if opt.branch3_weights_path:
        if opt.branch3_weights_path.endswith(".pth"):
            branch3.load_state_dict(torch.load(opt.branch3_weights_path))
            branches.append(branch3)


    ############## READ Clusters file and Create mapping ##########
    clusters = parse_clusters_config(opt.clusters_path)
    print(len(clusters))

    precision, recall, AP, f1, ap_class = evaluate_branches(
        backbone,
        branches,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        max_bound=opt.max_bound,
        clusters_path=opt.clusters_path,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
    print(AP.sum())
