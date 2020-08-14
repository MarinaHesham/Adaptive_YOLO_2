echo " >>>>>>>>>>>>>> 1. Train Original Network"
python3 ada_train.py --model_def config/yolov3-ada6.cfg --data_config config/coco.data --clusters_path clusters_sub.data --epochs 150
echo " >>>>>>>>>>>>>> 2. Train Classification Network"
python3 train_hier_classifier.py --model_def config/hier_clus.cfg --data_config config/coco.data --clusters_path clusters_sub.data --epochs 60
echo " >>>>>>>>>>>>>> 3.1 Test with cheating based Shuffling -- Higher bound"
python3 ada_test.py --model_def config/yolov3-ada6.cfg --data_config config/coco.data --clusters_path clusters_sub.data --weights_path checkpoints/yolov3_ada.pth --batch_size 1 --max_bound True
echo " >>>>>>>>>>>>>> 3.2 Test with Random Mode Shuffling -- Lower bound"
python3 ada_test.py --model_def config/yolov3-ada6.cfg --data_config config/coco.data --clusters_path clusters_sub.data --weights_path checkpoints/yolov3_ada.pth --batch_size 1
echo " >>>>>>>>>>>>>> 3.3 Test with Classification based Shuffling"
python3 ada_test.py --model_def config/yolov3-ada6.cfg --data_config config/coco.data --clusters_path clusters_sub.data --weights_path checkpoints/yolov3_ada.pth --batch_size 1 --hier_class True --hier_model_cfg config/hier_clus.cfg --hier_model checkpoints/yolov3_cluster_net.pth

echo " >>>>>>>>>>>>>> 5. Train Baseline Network"
python3 train.py --model_def config/yolov3-tiny6.cfg --data_config config/coco.data --epochs 150
echo " >>>>>>>>>>>>>> 6. Test Baseline Network"
python3 test.py --model_def config/yolov3-tiny6.cfg --data_config config/coco.data --weights_path checkpoints/yolov3_tiny.pth --batch_size 1
