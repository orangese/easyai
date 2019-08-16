#!/usr/bin/env bash

if [ ! -d "$HOME/coco" ] ; then
  echo "Preparing to download COCO. When prompted, please select all default options"
  mkdir coco
  cd coco || exit
  curl https://sdk.cloud.google.com | bash
  mkdir unlabeled2017
  gsutil -m rsync gs://images.cocodataset.org/unlabeled2017 unlabeled2017
  echo "Downloaded COCO to ~/coco"
else
  echo "COCO residing at ~/coco"
fi