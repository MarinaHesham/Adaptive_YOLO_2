python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 1 --ckpt_prefix 1 --epochs 60 > log_hier_1

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 2 --ckpt_prefix 2 --epochs 60 > log_hier_2

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 3 --ckpt_prefix 3 --epochs 60 > log_hier_3

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 4 --ckpt_prefix 4 --epochs 60 > log_hier_4

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 5 --ckpt_prefix 5 --epochs 60 > log_hier_5

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 6 --ckpt_prefix 6 --epochs 60 > log_hier_6

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 7 --ckpt_prefix 7 --epochs 60 > log_hier_7

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 8 --ckpt_prefix 8 --epochs 60 > log_hier_8

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 9 --ckpt_prefix 9 --epochs 60 > log_hier_9


python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 10 --ckpt_prefix 10 --epochs 60 > log_hier_10

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --pretrained_weights weights/yolov3_ada.weights --frozen_pretrained_layers 11 --ckpt_prefix 11 --epochs 60 > log_hier_11

python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --ckpt_prefix "separate" --epochs 60 > log_hier_separate
