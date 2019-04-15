#!/usr/bin/env bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# mode FGARCNN_cls_softmax_reg_offset, MaskRCNN_keypoint, RCNN_offset, RCNN_boundary_offset
source activate py3
#export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:$LD_LIBRARY_PATH
MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python3 ../train.py --train-path ../data/OCCLUSION/train.rec --val-path ../data/OCCLUSION/val.rec --network resnet50m --mode RCNN_boundary_offset --batch-size 8 --pretrain ../model/resnet-50 --epoch 0 --data-shape 512 --lr 0.001 --class-names 'obj_01, obj_05, obj_06, obj_08, obj_09, obj_10, obj_11, obj_12' --prefix ../output/OCCLUSION/rcnn-boundary-offset-resnet50m-512-stage3-7-size0.1-0.5-softcls-new/rcnn --gpu 0,1 --alpha-bb8 10.0 --end-epoch 36 --lr-steps '24,32' --wd 0.0005  --num-class 8

python3 ../evaluate.py --rec-path ../data/OCCLUSION/val.rec --network resnet50m --mode RCNN_boundary_offset --batch-size 32 --epoch 36 --data-shape 512 --class-names 'obj_01, obj_05, obj_06, obj_08, obj_09, obj_10, obj_11, obj_12' --prefix ../output/OCCLUSION/rcnn-boundary-offset-resnet50m-512-stage3-7-size0.1-0.5-softcls-new/rcnn --gpu 0 --num-class 8

