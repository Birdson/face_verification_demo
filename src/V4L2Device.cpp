#include "V4L2Device.h"

#include <string.h>

V4L2Device::V4L2Device() {
  v4l2sink = -1;
  vidsendsiz = 0;
}

V4L2Device::~V4L2Device() {
  if (v4l2sink > 0) {
    close(v4l2sink);
  }
}

bool V4L2Device::openV4L2Device(int width, int height, std::string virtaulCamDevice)
{
  this->size.width = width;
  this->size.height = height;
  this->virtaulCamDevice = virtaulCamDevice;
  v4l2sink = open(virtaulCamDevice.c_str(), O_WRONLY);
  if (v4l2sink < 0) {
    printf("Failed to open v4l2sink(%s) device.\n", virtaulCamDevice.c_str());
    return false;
  }
  // setup video for proper format
  struct v4l2_format v;
  int t;
  v.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  t = ioctl(v4l2sink, VIDIOC_G_FMT, &v);
  if( t < 0 )
    return false;

  v.fmt.pix.width = width;
  v.fmt.pix.height = height;

  bool writeYUVFormat = true;
  if (writeYUVFormat == false) {
    v.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    //v.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;

    vidsendsiz = width * height * 3;

    //vidsendsiz = (width * height * 3) >> 1;
    v.fmt.pix.sizeimage = vidsendsiz;
  } else {

    v.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;

    v.fmt.pix.field       = V4L2_FIELD_INTERLACED;
    /* Note VIDIOC_S_FMT may change width and height. */

    /* Buggy driver paranoia. */
    /*unsigned int min = v.fmt.pix.width * 2;
    if (v.fmt.pix.bytesperline < min)
      v.fmt.pix.bytesperline = min;
    min = v.fmt.pix.bytesperline * v.fmt.pix.height;
    if (v.fmt.pix.sizeimage < min)
      v.fmt.pix.sizeimage = min;
    vidsendsiz = v.fmt.pix.sizeimage;*/
    vidsendsiz = (width * height * 3) >> 1;
  }
  printf("openV4L2Device: device=%s, width=%d, height=%d, sendsize=%lu\n", virtaulCamDevice.c_str(), width, height, vidsendsiz);
  t = ioctl(v4l2sink, VIDIOC_S_FMT, &v);
  if( t < 0 )
    return false;

  //Add 1/30 fps parm
  struct v4l2_streamparm streamparm;
  memset(&streamparm, 0, sizeof(streamparm));
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  t = ioctl(v4l2sink, VIDIOC_G_PARM, &streamparm);
  if( t < 0 )
    return false;

  streamparm.parm.capture.capturemode |= V4L2_CAP_TIMEPERFRAME;
  streamparm.parm.capture.timeperframe.numerator = 1;
  streamparm.parm.capture.timeperframe.denominator = 30;
  t = ioctl(v4l2sink, VIDIOC_S_PARM, &streamparm);
  if( t < 0 )
    return false;

  return true;
}

bool V4L2Device::writeFrame(const void *frame) {
  bool result = true;
  if (vidsendsiz != write(v4l2sink, frame, vidsendsiz)) {
    printf("Write data to v4l2 device failed!\n");
    result = false;
  }
  return result;
}

std::string V4L2Device::getVirtaulCamDeviceName() {
  return virtaulCamDevice;
}

cv::Size V4L2Device::getSize() {
  return size;
}


