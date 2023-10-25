CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed9/seed10/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed10/seed12/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed12/seed13/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed13/seed14/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed14/seed17/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed17/seed18/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed18/seed24/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed24/seed25/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
sed -i 's/seed25/seed28/g'  configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/PASCAL_VOC/split3/5shot_CL_IoU.yml
