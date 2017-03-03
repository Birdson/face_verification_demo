#include "config_reader.h"
#include <stdio.h>

#include "inih/cpp/INIReader.h"

ConfigReader * ConfigReader::s_instance = NULL;

ConfigReader::ConfigReader() {

}

ConfigReader::~ConfigReader() {

}

ConfigReader* ConfigReader::getInstance() {
  if (s_instance == NULL)
    s_instance = new ConfigReader();
  return s_instance;
}

bool ConfigReader::initConfig() {
  INIReader reader(CONFIG_PATH);
  if (reader.ParseError() < 0) {
    printf("Can't load %s\n", CONFIG_PATH);
    return false;
  }

  webcam_config.device = reader.GetInteger("webcam", "device", 0);
  webcam_config.width = reader.GetInteger("webcam", "width", 1280);
  webcam_config.height = reader.GetInteger("webcam", "height", 720);
  webcam_config.enable_virtual_device = reader.GetBoolean("webcam", "enable_virtual_device", false);
  webcam_config.virtual_device_path = reader.Get("webcam", "virtual_device_path", "/dev/video1");

  test_config.enable_video_test = reader.GetBoolean("test", "enable_video_test", false);
  test_config.video_test_path = reader.Get("test", "video_test_path", "");
  test_config.enable_image_test = reader.GetBoolean("test", "enable_image_test", false);
  test_config.image_test_path = reader.Get("test", "image_test_path", "");

  cv_config.detection_framework = reader.GetInteger("cv", "framework", 0);
  cv_config.max_detection_num = reader.GetInteger("cv", "max_detection_num", 4);
  cv_config.max_detection_retry_num = reader.GetInteger("cv", "max_detection_retry_num", 5);
  cv_config.skip_frames = reader.GetInteger("cv", "skip_frames", 1);
  cv_config.face_registration_dir = reader.Get("cv", "face_registration_dir", "face_register/");
  cv_config.enable_check_blurry = reader.GetBoolean("cv", "enable_check_blurry", true);
  cv_config.enable_face_registration = reader.GetBoolean("cv", "enable_face_registration", false);
  cv_config.enable_draw_face_boxs = reader.GetBoolean("cv", "enable_draw_face_boxs", false);
  cv_config.enable_draw_face_landmarks = reader.GetBoolean("cv", "enable_draw_face_landmarks", false);
  cv_config.enable_draw_debug_information = reader.GetBoolean("cv", "enable_draw_debug_information", false);
  cv_config.enable_save_debug_face = reader.GetBoolean("cv", "enable_save_debug_face", false);

  yolo_config.enable = (cv_config.detection_framework == 0);
  yolo_config.model = reader.Get("yolo", "model", "");
  yolo_config.weight = reader.Get("yolo", "weight", "");
  yolo_config.sub_model = reader.Get("yolo", "sub_model", "");
  yolo_config.sub_weight = reader.Get("yolo", "sub_weight", "");
  yolo_config.confidence_threshold = reader.GetReal("yolo", "confidence_threshold", 0.24);
  yolo_config.sub_confidence_threshold = reader.GetReal("yolo", "sub_confidence_threshold", 0.24);

  ssd_config.enable = (cv_config.detection_framework == 1);
  ssd_config.model = reader.Get("ssd", "model", "");
  ssd_config.weight = reader.Get("ssd", "weight", "");
  ssd_config.confidence_threshold = reader.GetReal("yolo", "confidence_threshold", 0.4);

  opencv_config.enable = (cv_config.detection_framework == 2);
  opencv_config.model = reader.Get("opencv", "model", "");

  landmark_config.enable_caffe = reader.GetBoolean("landmark", "enable_caffe", true);
  landmark_config.caffe_model = reader.Get("landmark", "caffe_model", "");
  landmark_config.caffe_weight = reader.Get("landmark", "caffe_weight", "");
  landmark_config.caffe_mean = reader.Get("landmark", "caffe_mean", "");
  landmark_config.dlib_model = reader.Get("landmark", "dlib_model", "");

  sc_config.model = reader.Get("sc", "model", "");
  sc_config.weight = reader.Get("sc", "weight", "");
  sc_config.confidence_threshold = reader.GetReal("sc", "confidence_threshold", 0.5);
  sc_config.confidence_threshold_high = reader.GetReal("sc", "confidence_threshold_high", 0.5);

  return true;
}

void ConfigReader::showConfig() {

}
