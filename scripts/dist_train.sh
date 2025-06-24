#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-29500}

# 参数检查
if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "./scripts/dist_train.sh [number_of_gpus] [path_to_option_file]"
    exit 1
fi



# 启动命令
PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH}" \
torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    basicsr/train.py -opt "$CONFIG" --launcher pytorch "${@:3}"