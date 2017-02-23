#pragma once

#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <opencv2/core.hpp>

#include <unistd.h>     /* exit */
#include <stdio.h>
#include <string>

class V4L2Device {
public:
  V4L2Device();
  ~V4L2Device();
  bool openV4L2Device(int width, int height, std::string virtaulCamDevice);
  std::string getVirtaulCamDeviceName();
  bool writeFrame(const void *frame);
  cv::Size getSize();
private:
  int v4l2sink;
  ssize_t vidsendsiz;
  cv::Size size;

  std::string virtaulCamDevice;
};
