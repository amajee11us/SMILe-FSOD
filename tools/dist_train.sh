#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
ROI_LOSS=$3
PORT=${PORT:-29501}

if [[ -z "$ROI_LOSS" ]]; then
    export ROI_LOSS_TYPE="SupCon"
else
    export ROI_LOSS_TYPE=$3
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4}
