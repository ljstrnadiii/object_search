#!/bin/bash

# script to download pretrained models used

outdir="data/pretrained_models/"

if [[ ! -d $outdir ]]; then
    mkdir $outdir
fi

# get coco label map
wget -O $outdir'mscoco_complete_label_map.pbtxt' https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_complete_label_map.pbtxt

# download a model
model_path='http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz'
curl -SL $model_path | tar -xzf - -C $outdir
  
