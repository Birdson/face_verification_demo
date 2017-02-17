#pragma once

#define CONFIG_PATH "config.ini"

#include <string>

typedef struct {
  int device;
  int width;
  int height;
} WebCamConfig;

typedef struct {
  bool enable;
  std::string model;
  std::string weight;
  std::string sub_model;
  std::string sub_weight;
  float confidence_threshold;
  float sub_confidence_threshold;
} YoloConfig;

typedef struct {
  bool enable;
  std::string model;
  std::string weight;
  float confidence_threshold;
} SSDConfig;

typedef struct {
  bool enable;
  std::string model;
} HaarcascadeConfig;

typedef struct {
  bool enable_caffe;
  std::string caffe_model;
  std::string caffe_weight;
  std::string caffe_mean;
  std::string dlib_model;
} LandmarkConfig;

typedef struct {
  std::string model;
  std::string weight;
  float confidence_threshold;
} ScratchConfig;

typedef struct {
  bool enable_draw_face_boxs;
  bool enable_draw_face_landmarks;
  bool enable_draw_debug_information;
} DebugConfig;

class ConfigReader {
private:
  ConfigReader();
  static ConfigReader * s_instance;

public:
  ~ConfigReader();
  static ConfigReader* getInstance();
  bool initConfig();
  void showConfig();
  WebCamConfig webcam_config;
  YoloConfig yolo_config;
  SSDConfig ssd_config;
  HaarcascadeConfig haar_config;
  LandmarkConfig landmark_config;
  ScratchConfig sc_config;
  DebugConfig debug_config;
};
