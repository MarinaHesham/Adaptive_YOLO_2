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
from ptflops import get_model_complexity_info

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, max_bound=False, c_model=None):
    model.eval()
    common_classes = [0,1,2,7,13,16,24,25,26,27,39,41,45,56,58,60,67,73,74]
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    classification_time = datetime.timedelta(seconds=0)
    detection_time = datetime.timedelta(seconds=0)
    non_max_suppression_time = datetime.timedelta(seconds=0)

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    neglected_objects = 0
    all_objects = 1
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()

        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        
        # Select mode
        prev_time = time.time()
        if max_bound:
            # print(targets[:, 1].tolist())
            ts = targets[:, 1].tolist()
            cluster_cnt = np.zeros(len(model.mode_classes_list))
            for t in ts:
                for i, cluster in enumerate(model.mode_classes_list):
                    if t in cluster and t not in common_classes:
                        cluster_cnt[i] += 1
            # print("Cluster Count", cluster_cnt, "Chosen Max", np.argmax(cluster_cnt))
            neglected_objects += np.sum(cluster_cnt) - np.max(cluster_cnt)
            all_objects += np.sum(cluster_cnt)
            # print("Neglected/All Objects = ", neglected_objects, "/", all_objects)

            model.mode = np.argmax(cluster_cnt)
        else:
            model.mode = random.randint(0, 1)
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        classification_time += inference_time

        with torch.no_grad():
            prev_time = time.time()
            outputs = model(imgs)

            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            detection_time += inference_time

            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            non_max_suppression_time += inference_time
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    
    print("Neglected/All Objects = ", 100.0*neglected_objects/all_objects)

    print("Classification time is ", classification_time, "Average per image is ", (classification_time/len(dataloader)).microseconds/1000, "ms")
    print("Detection time is ", detection_time, "Average per image is ", (detection_time/len(dataloader)).microseconds/1000, "ms")
    print("NMS time is ", non_max_suppression_time, "Average per image is ", (non_max_suppression_time/len(dataloader)).microseconds/1000, "ms")
    print("Total Inference time is ", classification_time+detection_time+non_max_suppression_time, "Average per image is ", ((classification_time+detection_time+non_max_suppression_time)/len(dataloader)).microseconds/1000, "ms")

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
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
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
        for i, element in enumerate(cluster):
            class_to_cluster[element] = i

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
        classify_model = ClusterModel(opt.hier_model_cfg, len(clusters)).to(device)
        classify_model.apply(weights_init_normal)
        classify_model.load_state_dict(torch.load(opt.hier_model))

        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(classify_model, (3, 416, 416), as_strings=True, print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print("Compute mAP...")
    model.shared_layers = int(opt.hier_model_shared_layers)
    model.classification_model = classify_model
    if classify_model is not None:
        model.classification_model.shared_layers = int(opt.hier_model_shared_layers)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 416, 416), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
   
    if opt.iou_thres == -1:
        ap5_95 = []
        for iou_thres in np.arange(0.5,0.95,0.05):
            print("Evaluating at IOU_thres = ", iou_thres)
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                max_bound=opt.max_bound,
            )

            print("Average Precisions:")
            for i, c in enumerate(ap_class):
                print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

            print(f"mAP: {AP.mean()}")
            ap5_95.append(AP.mean())
        print(ap5_95, "mAP(0.5:0.95) = ", sum(ap5_95)/len(ap5_95))
    elif opt.nms_thres == -1:
        ap5_95 = []
        for nms_thres in np.arange(0.1,0.75,0.05):
            print("Evaluating at nms thresh = ", nms_thres)
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                nms_thres=nms_thres,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                max_bound=opt.max_bound,
            )

            print("Average Precisions:")
            for i, c in enumerate(ap_class):
                print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

            print(f"mAP: {AP.mean()}")
            ap5_95.append(AP.mean())
        print(ap5_95, "mAP(0.1:0.755) = ", sum(ap5_95)/len(ap5_95))        
    else:
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=opt.batch_size,
            max_bound=opt.max_bound,
        )

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")
        print(AP.sum())
