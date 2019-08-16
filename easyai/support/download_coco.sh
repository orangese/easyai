#!/usr/bin/env bash

if [ ! -d $~/coco ]; then
  echo "Preparing to download COCO. When prompted, please select all default options"
  mkdir coco
  cd coco || exit
  curl https://sdk.cloud.google.com | bash
  mkdir train2017
  gsutil -m rsync gs://images.cocodataset.org/train2017 train2017
  echo "Downloaded COCO to ~/coco"
else
  :
fi