#!/usr/bin/env bash
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source activate py3
python3 ../evaluate.py --rec-path ../data/OCCLUSION/val.rec --network resnet50m --mode RCNN_boundary_offset --batch-size 32 --epoch 36 --data-shape 512 --class-names 'obj_01, obj_05, obj_06, obj_08, obj_09, obj_10, obj_11, obj_12' --prefix ../output/OCCLUSION/rcnn-boundary-offset-resnet50m-512-stage3-7-size0.1-0.5/rcnn --gpu 0 --num-class 8


