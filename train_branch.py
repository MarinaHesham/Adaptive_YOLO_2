from __future__ import division

from ada_models import *
from models import Darknet
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from ada_test import evaluate, evaluate_branch

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch, torch.nn as nn

from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from ptflops import get_model_complexity_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradient accums before step")
    parser.add_argument("--branch_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--backbone_def", type=str, default="config/yolov3-tiny.cfg", help="path to backbone model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--backbone_weights", type=str, help="backbone model weights, must be specified")
    parser.add_argument("--backbone_num_of_layers", type=str, help="number of layers in backbone model, must be specified")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--frozen_pretrained_layers", type = int, default=-1, help="number of the front layers that should be loaded from pretrained weights")
    parser.add_argument("--clusters_path", type=str, default="clusters.data", help="clusters file path")
    parser.add_argument("--cluster_index", type=int, default=0, help="Branch cluster index")
    parser.add_argument("--ckpt_prefix", type=str, default="", help="pre for checkpoints files")

    opt = parser.parse_args()
    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    num_classes = int(data_config["classes"])

    # 1. Create Backbone Model
    backbone = Backbone(opt.backbone_def).to(device)
    print("Bachbone Number of Parameters")
    count_parameters(backbone)
    
    # 2. Load Backbone Weights
    if opt.backbone_weights:
        backbone.load_darknet_weights(opt.backbone_weights, int(opt.backbone_num_of_layers))

    # 3. Create Branch Model
    branch = Darknet(opt.branch_def).to(device)
    print("Branch Number of Parameters")
    count_parameters(branch)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            branch.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            branch.load_darknet_weights(opt.pretrained_weights,opt.frozen_pretrained_layers)

    branch.apply(weights_init_normal)
    
    optimizer = torch.optim.Adam(branch.parameters())

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

    ############## READ Clusters file and Create mapping ##########
    clusters = parse_clusters_config(opt.clusters_path)
    class_to_cluster_list = get_class_to_cluster_map(clusters)

    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(branch, (3, 416, 416), as_strings=True, print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    best_model = branch
    best_map = 0

    for epoch in range(opt.epochs):
        dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        branch.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # 4. Filter data and Convert labels
            # Convert the original labels to internal cluster based labels
            targets = map_labels_to_cluster(targets, clusters, class_to_cluster_list, opt.cluster_index, device)
            if targets.shape[0] == 0:
                continue

            # 5. Execute Backbone
            backbone_out = backbone(imgs)

            # 6. pass output of backbone to branch model for training
            loss, outputs = branch(backbone_out, targets)

            if outputs == None:
                continue
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Cluster %d, Epoch %d/%d, Batch %d/%d] ----\n" % (opt.cluster_index, epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(branch.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in branch.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(branch.yolo_layers):
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

            branch.seen += imgs.size(0)

        torch.save(branch.state_dict(), f"checkpoints/%s_yolov3_ckpt_clus%d_%d.pth" %(opt.ckpt_prefix, opt.cluster_index, epoch))

        print(f"\n---- Evaluating Model on Cluster ----", opt.cluster_index)
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = evaluate_branch(
            backbone,
            branch,
            path=valid_path,
            iou_thres=0.5,
            conf_thres=0.3,
            nms_thres=0.5,
            img_size=opt.img_size,
            batch_size=1,
            clusters_path=opt.clusters_path,
            clusters_idx=opt.cluster_index,
            device=device
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
        if AP.mean() > best_map:
            best_map = AP.mean()
            best_model = branch

    print("Saving best model of mAP", best_map)

    best_model.save_darknet_weights("weights/%s_yolov3_ada.weights" % opt.ckpt_prefix)
    torch.save(best_model.state_dict(), f"checkpoints/%s_yolov3_ada.pth" % opt.ckpt_prefix)
