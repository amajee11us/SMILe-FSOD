CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
sed -i 's/seed13/seed14/g'  configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
sed -i 's/seed14/seed17/g'  configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
sed -i 's/seed17/seed18/g'  configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
sed -i 's/seed18/seed24/g'  configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
sed -i 's/seed24/seed25/g'  configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/PASCAL_VOC/split1/5shot_CL_IoU.yml
