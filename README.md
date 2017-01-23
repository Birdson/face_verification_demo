# CNN based Face Verification Demo

## Requirements
1. [OpenCV](https://github.com/opencv/opencv)
2. [Caffe SSD](https://github.com/weiliu89/caffe/tree/ssd)
3. Download models and unzip it to root


## Build
1. Modify "CAFFE_ROOT" to point to your Caffe SSD
2. mkdir build
3. cd build
4. cmake ..
5. make -j8

## Run
./build/face_verification_demo
