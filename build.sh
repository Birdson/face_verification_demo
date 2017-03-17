#!/bin/bash

ln -sf ${PWD}/external/yolo/lib/libdarknet.so.0 ${PWD}/external/yolo/lib/libdarknet.so

ln -sf ${PWD}/external/caffe/lib/libcaffe.so.1.0.0-rc3 ${PWD}/external/caffe/lib/libcaffe.so

rm -r build
mkdir build
cd build
cmake ..
make -j8

cd ..
