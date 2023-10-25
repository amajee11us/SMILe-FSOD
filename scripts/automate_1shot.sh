declare -a seeds=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
for s in "${seeds[@]}"
do
    sed -i "s/seed[0-9]/seed"$s"/g" configs/PASCAL_VOC/split1/1shot_baseline.yml
    CUDA_VISIBLE_DEVICES=3 python tools/train_net.py --config-file configs/PASCAL_VOC/split1/1shot_baseline.yml
done