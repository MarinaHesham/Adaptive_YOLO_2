from __future__ import division

from ada_models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from classify import *
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

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
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
    num_classes = int(data_config["classes"])


    # Initiate model
    model = ClusterModel(opt.model_def).to(device)
    model.num_all_classes = num_classes

    clusters = parse_clusters_config(opt.clusters_path)

    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights,opt.frozen_pretrained_layers)

    # Freeze the loaded layers
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < opt.frozen_pretrained_layers:
            print("Freeze ", name, " ", i)
            param.requires_grad = False

    count_parameters(model)

    dataset = ClustersDataset(train_path, augment=True, multiscale=opt.multiscale_training,  clusters=clusters)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            optimizer.zero_grad()   # zero the gradient buffers
            output = model(imgs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()    # Does the update
            
            model.seen += imgs.size(0)
            if batch_i % 100 == 0:
                print(batch_i,len(dataloader), loss.cpu().detach().numpy())

        torch.save(model.state_dict(), f"checkpoints/yolov3_cluster_net_%d.pth" % epoch)

        dataset_valid = ClustersDataset(valid_path, augment=True, multiscale=False,  clusters=clusters)

        dataloader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=dataset_valid.collate_fn
        )

        print("Evaluate on ", len(dataloader_valid))

        right_predictions = 0
        all_predictions = 0
        
        for batch_i, (_, imgs, targets) in enumerate(dataloader_valid):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            output = model(imgs)
            loss = criterion(output, targets)

            output = torch.argmax(output.cpu(), dim=1).numpy()
            targets = torch.argmax(targets.cpu(), dim=1).numpy()
            right_predictions += np.count_nonzero(targets==output)
            all_predictions += len(targets)

        print("Accuracy = ", 100.0*right_predictions/all_predictions)
