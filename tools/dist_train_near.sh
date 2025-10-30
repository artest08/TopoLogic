#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/topologic_near
CONFIG=projects/configs/topologic_r50_8x1_24e_olv2_subset_A_near.py

GPUS=$1
PORT=${PORT:-28510}

mkdir -p ${WORK_DIR}

~/containers/python_topomlp -m torch.distributed.run  --nproc_per_node=$GPUS --master_port=$PORT\
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log

