#!/usr/bin/env bash

if [ ! -d "$HOME/coco" ] ; then
  echo "Preparing to download COCO. When prompted, please select all default options"
  sleep 2s
  mkdir coco
  cd coco || echo "Local directory 'coco' was not found" ; exit
  curl https://sdk.cloud.google.com || echo "https://sdk.cloud.google.com is not accesible" ; exit
  bash --login
  mkdir unlabeled2017
  gsutil -m rsync gs://images.cocodataset.org/unlabeled2017 unlabeled2017 || echo "Failed to download images" ; exit
  echo "Downloaded COCO to ~/coco"
else
  # redundancy check for existence of COCO
  echo "COCO residing at ~/coco"
fi