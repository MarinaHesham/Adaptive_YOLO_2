from __future__ import division

import os
import sys
import argparse

import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type=str, default="data/coco/trainvalno5k.txt", help="path to data config file")
    parser.add_argument("--output_file_path", type=str, default="data/coco/trainvalno5k_subset.txt", help="path to subset data config file")

    opt = parser.parse_args()

    with open(opt.data_file_path, "r") as file:
        img_files = file.readlines()
    
    label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        for path in img_files]
    
    #print(img_files)
    
    wanted_classes = [2,6,5,69,72,57]
    wanted_classes_counter = np.zeros(len(wanted_classes))

    filtered_img_paths = []

    for i_file, l_file in enumerate(label_files):
        keep = False
        l_file = l_file.rstrip()
        if os.path.exists(l_file):
            boxes = np.loadtxt(l_file).reshape(-1, 5)
            for i, label in enumerate(boxes[:,0]):
                if label in wanted_classes:
                    if wanted_classes_counter[wanted_classes.index(int(label))] < 3000:
                        keep = True
                        wanted_classes_counter[wanted_classes.index(int(label))] += 1

                if keep == True:
                    filtered_img_paths.append(i_file)
                    modified_label_path = l_file.replace("labels", "labels_sub")
                    boxes = np.asarray([box for box in boxes if box[0] in wanted_classes])
                    boxes[:,0] = [wanted_classes.index(boxes[i,0]) for i in range(len(boxes))] 
                    np.savetxt(modified_label_path, boxes, delimiter=" ", fmt=['%d','%.6f','%.6f','%.6f','%.6f'])

                    break
        
    filtered_img_paths = [img_files[i] for i in filtered_img_paths]
    
    print(wanted_classes_counter)

    with open(opt.output_file_path, 'w') as filehandle:
        filehandle.writelines("%s\n" % img.rstrip() for img in filtered_img_paths)
