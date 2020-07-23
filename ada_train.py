from __future__ import division

from ada_models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from ada_test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny.weights", help="path to weights file")
    parser.add_argument("--frozen_pretrained_layers", type = int, default=9, help="number of the front layers that should be loaded from pretrained weights")
    parser.add_argument("--clusters_path", type=str, default="clusters.data", help="clusters file path")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = AdaptiveYOLO(opt.model_def).to(device)
    count_parameters(model)
    
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

    ## Set the clusters and cluster mapping for the model
    model.mode_dicts_class_to_cluster = class_to_cluster_list
    model.mode_classes_list = clusters

    model.apply(weights_init_normal)
    
    # Load first the front layers weights 
    model.load_darknet_weights(opt.weights_path, opt.frozen_pretrained_layers)
    
    # Freeze the loaded layers
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if i < cutoff:
    #         print("Freeze ", name, " ", i)
    #         param.requires_grad = False

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)


    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    ########### Train for each cluster ############

    for mode_i in range(len(clusters)):
        ####### 1. Prepare cluster data #######
        dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        ####### 1. Start training #######
        # ## Freeze cluster 2 parameters 
        # c2_start = 25

        # # for i, (name, param) in enumerate(model.named_parameters()):
        # #     if i >= c2_start:
        # #         print("Freeze ", name, " ", i)
        # #         param.requires_grad = False

        model.mode = mode_i
        
        for epoch in range(opt.epochs):
            model.train()
            start_time = time.time()
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                batches_done = len(dataloader) * epoch + batch_i

                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)

                loss, outputs = model(imgs, targets)

                if outputs == None:
                    continue
                loss.backward()

                if batches_done % opt.gradient_accumulations:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                # ----------------
                #   Log progress
                # ----------------

                log_str = "\n---- [Cluster %d, Epoch %d/%d, Batch %d/%d] ----\n" % (mode_i, epoch, opt.epochs, batch_i, len(dataloader))

                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

                model.seen += imgs.size(0)

                if batch_i % opt.checkpoint_interval == 0:
                    torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_clus%d_%d.pth" % (mode_i, batch_i))
        
        print(f"\n---- Evaluating Model on Cluster ----", mode_i)
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=0.5,
            conf_thres=0.2,
            nms_thres=0.5,
            img_size=opt.img_size,
            batch_size=8,
        )
        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean()}")

        torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_clus%d_%d.pth" %(mode_i, epoch))

    torch.save(model.state_dict(), "checkpoints/yolov3_ada.pth")
