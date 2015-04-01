#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

../../build/tools/compute_image_mean /home/titan/remote_pc/IMAGENET_2k_train_leveldb \
  /home/titan/Devel/caffe/data/imagenet_getty_2000/imagenet_mean.binaryproto leveldb

echo "Done."
