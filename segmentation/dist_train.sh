#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29300}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# bash dist_train.sh configs/cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.py 2 --resume-from work_dirs/upernet_internimage_b_512x1024_160k_cityscapes/latest.pth
# python test.py work_dirs/upernet_internimage_b_512x1024_160k_cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.py work_dirs/upernet_internimage_b_512x1024_160k_cityscapes/latest.pth --show-dir visualization --out work_dirs/format_results