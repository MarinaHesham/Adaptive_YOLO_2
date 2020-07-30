from __future__ import division

from ada_models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

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

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, c_model=None):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        if c_model != None:
            model.mode = torch.argmax(c_model(imgs), dim=1).numpy()[0] 
        else:
            model.mode = random.randint(0, 1)
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--clusters_path", type=str, default="clusters.data", help="clusters file path")
    parser.add_argument("--hier_class", type=bool, default=False, help="when True enable hierarical classification")
    parser.add_argument("--hier_model_cfg", type=str, help="path to hierarical model congiguration")
    parser.add_argument("--hier_model", type=str, help="path to hierarical classification model")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
 
    # Initiate model
    model = AdaptiveYOLO(opt.model_def).to(device)
    count_parameters(model)
    num_classes = int(data_config["classes"])

    # Initiate model
    model.num_all_classes = num_classes

    ############## READ Clusters file and Create mapping ##########
    clusters = parse_clusters_config(opt.clusters_path)
    print(len(clusters))
    class_to_cluster_list = []

    ## create the class-cluster map to be used for labels in split training
    for cluster in clusters:
        class_to_cluster = {}
        cluster_to_class = {}
        for i, element in enumerate(cluster):
            class_to_cluster[element] = i
            cluster_to_class[i] = element

        class_to_cluster_list.append(class_to_cluster)

    model.mode_dicts_class_to_cluster = class_to_cluster_list
    model.mode_classes_list = clusters

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    classify_model = None
    if opt.hier_class:
        classify_model = ClusterModel(opt.hier_model_cfg)
        classify_model.num_all_classes = num_classes

        classify_model.apply(weights_init_normal)

        # If specified we start from checkpoint
        classify_model.load_state_dict(torch.load(opt.hier_model))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        c_model=classify_model,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
