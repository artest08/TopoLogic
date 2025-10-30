#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/topologic_test
CONFIG=projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py

CHECKPOINT=work_dirs/topologic_r50_8x1_24e_olv2_subset_A.pth

GPUS=$1
PORT=${PORT:-28511}

~/containers/python_topomlp -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
