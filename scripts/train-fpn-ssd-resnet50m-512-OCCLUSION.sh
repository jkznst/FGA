#!/usr/bin/env bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source activate py3
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:$LD_LIBRARY_PATH
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 ../train.py --train-path ../data/OCCLUSION/train.rec --val-path ../data/OCCLUSION/val.rec --network resnet50m --mode RCNN_offset --batch-size 4 --pretrain ../model/resnet-50 --epoch 0 --data-shape 800 --lr 0.001 --class-names 'obj_01, obj_05, obj_06, obj_08, obj_09, obj_10, obj_11, obj_12' --prefix ../output/OCCLUSION/rcnn-bb8offset-resnet50m-800-stage3-7-size0.1-0.5/rcnn --gpu 2,3 --alpha-bb8 10.0 --end-epoch 45 --lr-steps '30,40' --wd 0.0005  --num-class 8

