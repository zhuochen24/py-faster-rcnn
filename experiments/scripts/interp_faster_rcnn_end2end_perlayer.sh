#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc tmp_train_val/xxx.prototxt\
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
TEST_PROTO=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${TEST_PROTO}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#time ./tools/train_net.py --gpu ${GPU_ID} \
#  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/interp_conv43_solver_trainall.prototxt \
#   --weights data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel \
#  --imdb ${TRAIN_IMDB} \
#  --iters ${ITERS} \
#  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
#  ${EXTRA_ARGS}
#
#set +x
#NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
#set -x

#_lininterp_conv11.prototxt \
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/tmp_train_val/${TEST_PROTO} \
  --net data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}

 # --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test_lininterp_conv11.prototxt \
 #  --net data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel \
 # --net ${NET_FINAL} \
