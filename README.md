# CNN based Face Verification Demo

## Requirements
1. [OpenCV](https://github.com/opencv/opencv)
2. [Caffe SSD](https://github.com/weiliu89/caffe/tree/ssd)
Download the Caffe SSD, then follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

3. Download the models from [Dropbox](https://www.dropbox.com/s/cdyd8bm0bzzwxz1/models.tar?dl=0)
and stored models in the root of Face Verification Demo.
  ```Shell
  # Extract the models.
  tar -xvf models.tar
  ```

## Build
  ```Shell
  # Modify `CAFFE_ROOT` in CMakeLists.txt to point to your Caffe SSD
  mkdir build
  cd build
  cmake ..
  make -j8
  ```

## Run
  ```Shell
  ./build/face_verification_demo
  ```
