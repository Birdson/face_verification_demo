/*
 * Copyright 2009-2016 Pegatron Corporation. All Rights Reserved.
 *
 * Pegatron Corporation. Confidential and Proprietary
 *
 * The following software source code ("Software") is strictly confidential and
 * is proprietary to Pegatron Corporation. ("PEGATRON").  It may only be read,
 * used, copied, adapted, modified or otherwise dealt with by you if you have
 * entered into a confidentiality agreement with PEGATRON and then subject to the
 * terms of that confidentiality agreement and any other applicable agreement
 * between you and PEGATRON.  If you are in any doubt as to whether you are
 * entitled to access, read, use, copy, adapt, modify or otherwise deal with
 * the Software or whether you are entitled to disclose the Software to any
 * other person you should contact PEGATRON.  If you have not entered into a
 * confidentiality agreement with PEGATRON granting access to this Software you
 * should forthwith return all media, copies and printed listings containing
 * the Software to PEGATRON.
 *
 * PEGATRON reserves the right to take legal action against you should you breach
 * the above provisions.
 *
 ******************************************************************************/
#include "config_reader.h"
#include "cv/face_verification.h"

#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dlib/data_io.h"
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/error.h>
#include <dlib/image_transforms.h>

#include <boost/format.hpp>
#include <boost/lambda/bind.hpp>

#include <omp.h>

#define DEBUG 1

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace boost::lambda;

static double detect_time = 0, alignment_time = 0, predit_time = 0;
static const std::string CV_ROOT_DIR = "./";
static const std::string CV_FACE_REGISTER_DIR = CV_ROOT_DIR + "face_register/";
static const std::string CV_FACE_PREDICT_DIR= CV_ROOT_DIR + "face_predict/";
static const std::string CV_TEMP_DIR = CV_ROOT_DIR + ".temp/";

static const std::string POSE_NAME[] = {"Pitch", "Yaw", "Roll"};

inline float getIoU(Rect a, Rect b)
{
  Rect unionRect = a | b;
  Rect intersectionRect = a & b;
  float iou = (float)intersectionRect.area()/(float)unionRect.area();
  return iou;
}

inline void CVRect_to_DlibRect(vector<dlib::rectangle>& d_rect, vector<Rect>& cv_rect)
{
  dlib::rectangle rect;

  d_rect.clear();

  for(unsigned int i=0; i < cv_rect.size(); i++)
  {
    rect.set_left(cv_rect[i].x);
    rect.set_top(cv_rect[i].y);
    rect.set_right(cv_rect[i].x + cv_rect[i].height); //x + height
    rect.set_bottom(cv_rect[i].y + cv_rect[i].width); // y + width
    d_rect.push_back(rect);
  }
}

FaceVerification::FaceVerification()
{
  cfv_ = NULL;
  enable_auto_face_registration = true;
  enable_auto_face_registration_retry = true && enable_auto_face_registration;
  if (!boost::filesystem::exists(CV_TEMP_DIR)) {
    boost::filesystem::create_directory(CV_TEMP_DIR);
  }
  init_libs_state_ = init();
  if (init_libs_state_) loadRegisteredFaces();
}

FaceVerification::~FaceVerification()
{
  delete fv_result;
  delete cfd_;
  delete cfld_;
  delete cfv_;
  free(face_boxes);
}

bool FaceVerification::init(void)
{
  bool init_state = false;
  if (cascade.empty() || cfd_ == NULL || cfv_ == NULL) {
    printf("Initial Face CV Libs ...\n");
    if (initFaceDetection() && initFaceLandmarkDetection() && initFaceVerification()) {
      printf("Initial Face CV Libs Success!\n");
      init_state = true;
    } else {
      printf("Initial Face CV Libs Fail!\n");
    }
  } else {
    init_state = true;
  }
  return init_state;
}

void FaceVerification::loadRegisteredFaces(void)
{
  face_register_paths.clear();
  face_register_features.clear();
  retry_face_register_features.clear();
  if (!boost::filesystem::exists(CV_FACE_REGISTER_DIR)) {
    boost::filesystem::create_directory(CV_FACE_REGISTER_DIR);
  } else {
    directory_iterator dirIter(CV_FACE_REGISTER_DIR);
    directory_iterator endIter;
    for( ; dirIter != endIter; ++dirIter ) {
      try {
        if(is_directory(dirIter->path())) {
          face_register_paths.push_back(dirIter->path());
          std::vector<dlib::file> face_register_files = dlib::get_files_in_directory_tree(dirIter->path().string(),
              dlib::match_endings(".png .PNG .jpeg .JPEG .jpg .JPG"), 10);
          float* face_register_feature = cfv_->extract_feature(face_register_files[0].full_name());
          face_register_features.push_back(face_register_feature);
          if (face_register_files.size() > 1) {
              face_register_feature = cfv_->extract_feature(face_register_files[1].full_name());
          } else {
              face_register_feature = NULL;
          }
              retry_face_register_features.push_back(face_register_feature);
          }
        } catch(std::exception const& ex) {
        cerr << dirIter->path() << " " << ex.what() << endl;
      }
    }
  }
}

bool FaceVerification::initFaceDetection(void)
{

  bool init_state = false;
  try
  {
    double costtime;
    clock_t time = clock();

    if (ConfigReader::getInstance()->yolo_config.enable)
    {
      const string model = ConfigReader::getInstance()->yolo_config.model;
      const string weight = ConfigReader::getInstance()->yolo_config.weight;
      const string sub_model = ConfigReader::getInstance()->yolo_config.sub_model;
      const string sub_weight = ConfigReader::getInstance()->yolo_config.sub_weight;

      //Using sub yolo face detection to check face exits before face registration
      if (!boost::filesystem::exists(model) && !boost::filesystem::exists(sub_model)) {
        CHECK_GT(sub_model.size(), 0) << "Need a Face Detection Model to detect face.";
        printf("Can't find Face Detection Model %s\n", sub_model.c_str());
        return false;
      }
      if (!boost::filesystem::exists(sub_weight) && !boost::filesystem::exists(weight)) {
        CHECK_GT(sub_weight.size(), 0) << "Need a Face Detection Weight to detect face.";
        printf("Can't find Face Detection Weight %s\n", sub_weight.c_str());
        return false;
      }

      init_state = yolo_init((char *)model.c_str(), (char *)weight.c_str())
            && sub_yolo_init((char *)sub_model.c_str(), (char *)sub_weight.c_str());
    }
    else if (ConfigReader::getInstance()->ssd_config.enable)
    {
      const string model = ConfigReader::getInstance()->ssd_config.model;
      const string weight = ConfigReader::getInstance()->ssd_config.weight;

      if (!boost::filesystem::exists(model)) {
        CHECK_GT(model.size(), 0) << "Need a Face Detection Model to detect face.";
        printf("Can't find Face Detection Model %s\n", model.c_str());
        return false;
      }

      if (!boost::filesystem::exists(weight)) {
        CHECK_GT(weight.size(), 0) << "Need a Face Detection Weight to detect face.";
        printf("Can't find Face Detection Weight %s\n", weight.c_str());
        return false;
      }

      const string& mean_value = "104,117,123";
      cfd_ = new caffe::CaffeFaceDetection(model, weight, "", mean_value);
      init_state = true;
    }
    else if (ConfigReader::getInstance()->haar_config.enable)
    {
      const std::string model = ConfigReader::getInstance()->haar_config.model;
      if (!boost::filesystem::exists(model)) {
        CHECK_GT(model.size(), 0) << "Need a Face Detection Model to detect face.";
        printf("Can't find Face Detection Model %s\n", model.c_str());
        return false;
      }
      init_state = cascade.load(model);
    }

    costtime = 1000.0 * (clock()-time)/(double) CLOCKS_PER_SEC;
#if DEBUG
    printf("Load Face Detection time is %f ms\n", costtime);
#endif
  }
  catch(cv::Exception& e)
  {
    printf("initialLib caught cv::Exception: %s\n", e.what());
  }
  catch (...)
  {
    printf("dlibInitial caught unknown exception\n");
  }
  return init_state;
}

bool FaceVerification::initFaceLandmarkDetection(void)
{
  bool init_state = false;
  try
  {
    double costtime;
    clock_t time = clock();
    if(ConfigReader::getInstance()->landmark_config.enable_caffe)
    {
      const string caffe_model = ConfigReader::getInstance()->landmark_config.caffe_model;
      const string caffe_weight = ConfigReader::getInstance()->landmark_config.caffe_weight;
      const string caffe_mean = ConfigReader::getInstance()->landmark_config.caffe_mean;
      if (!boost::filesystem::exists(caffe_model))  {
          CHECK_GT(caffe_model.size(), 0) << "Need a Face Landmark Model to detect face.";
          printf("Can't find Face Landmark Model %s\n", caffe_model.c_str());
          return false;
      }
      if (!boost::filesystem::exists(caffe_weight))  {
          CHECK_GT(caffe_weight.size(), 0) << "Need a Face Landmark Weight to detect face.";
          printf("Can't find Face Landmark Weight %s\n", caffe_weight.c_str());
          return false;
      }
      if (!boost::filesystem::exists(caffe_mean))  {
          CHECK_GT(caffe_mean.size(), 0) << "Need a Face Landmark Mean to detect face.";
          printf("Can't find Face Landmark Mean %s\n", caffe_mean.c_str());
          return false;
      }

      cfld_ = new caffe::CaffeFaceLandmarkDetection(caffe_model, caffe_weight, caffe_mean);
    }
    else
    {
      const string dlib_model = ConfigReader::getInstance()->landmark_config.dlib_model;
      if (!boost::filesystem::exists(dlib_model))  {
          CHECK_GT(dlib_model.size(), 0) << "Need a Face Landmark Model to detect face.";
          printf("Can't find Face Landmark Model %s\n", dlib_model.c_str());
          return false;
      }
      dlib::deserialize(dlib_model) >> pose_model;
    }
    costtime = 1000.0 * (clock()-time)/(double) CLOCKS_PER_SEC;
#if DEBUG
    printf("Load Pose Model time is %f ms\n", costtime);
#endif
    init_state = true;
  }
  catch(dlib::error &err)
  {
    printf("initialLib caught dlib::error: %s\n", err.what());
  }
  catch (...)
  {
    printf("dlibInitial caught unknown exception\n");
  }
  return init_state;
}

bool FaceVerification::initFaceVerification(void)
{
  const string model = ConfigReader::getInstance()->sc_config.model;
  const string weight = ConfigReader::getInstance()->sc_config.weight;
  if (!boost::filesystem::exists(model))  {
    CHECK_GT(model.size(), 0) << "Need a Model Definition File to verify face.";
    printf("Can't find Model Definition %s\n", model.c_str());
    return false;
  }
  if (!boost::filesystem::exists(weight))  {
    CHECK_GT(weight.size(), 0) << "Need a Model Definition File to verify face.";
    printf("Can't find Model Definition %s\n", weight.c_str());
    return false;
  }
#if DEBUG
  printf("Initial Cafffe Model Definition: %s\n", model.c_str());
  printf("Initial Cafffe Model Weights: %s\n", weight.c_str());
#endif
  double costtime;
  clock_t time = clock();
  cfv_ = new caffe::CaffeFaceVerification(model, weight);
  costtime = 1000.0 * (clock()-time)/(double) CLOCKS_PER_SEC;
#if DEBUG
  printf("Load Caffe Net time is %f ms\n", costtime);
#endif
  return true;
}

void FaceVerification::faceDetection(Mat& img, vector<Rect>& faces)
{
  Mat gray;
  cvtColor( img, gray, CV_BGR2GRAY );
  clock_t time = clock();
  if (ConfigReader::getInstance()->yolo_config.enable) {
    IplImage *frame = new IplImage(img);
    const float threshold = ConfigReader::getInstance()->yolo_config.confidence_threshold;
    face_boxes = yolo_detect(frame, threshold, .5, 1.2);
    if (face_boxes != NULL) {
      int detection_num = yolo_get_detection_num();
      for (int i = 0; i < detection_num; ++i) {
        if (face_boxes[i].width > 0 && face_boxes[i].height > 0) {
          faces.push_back(Rect(face_boxes[i].xmin, face_boxes[i].ymin, face_boxes[i].width, face_boxes[i].height));
        }
      }
    }
  }
  else if (ConfigReader::getInstance()->ssd_config.enable)
  {
    const cv::Size min_face_detect_size(50, 50);
    const float threshold = ConfigReader::getInstance()->ssd_config.confidence_threshold;
    std::vector<std::vector<float>> detections = cfd_->Detect(img);
    const int detection_size = 7;
    for (int i = 0; i < detections.size(); ++i) {
      const std::vector <float> detection = detections[i];
      // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].

      CHECK_EQ(detection.size(), detection_size);

      const float score = detection[2];
      if (score >= threshold) {
        int xmin = max(static_cast<int>(detection[3] * img.cols), 0);
        int ymin = max(static_cast<int>(detection[4] * img.rows), 0);
        int xmax = min(static_cast<int>(detection[5] * img.cols), img.cols);
        int ymax = min(static_cast<int>(detection[6] * img.rows), img.rows);
        int width = xmax - xmin;
        int height = ymax - ymin;
        const int x_center = xmin + width/2;
        const int y_center = ymin + height/2;
        xmin = max(x_center - max(width, height)/2, 0);
        ymin = max(y_center - max(width, height)/2, 0);
        width = max(width, height);
        height = max(width, height);
        if (xmin + width > img.cols)
          width = img.cols - xmin;
        if (ymin + height > img.rows)
          height = img.rows - ymin;

        if (min(width, height) > min_face_detect_size.width) {
          const float ratio = ((float)min(width, height)/(float)max(width, height));
          if (ratio > 0.65) {
            faces.push_back(Rect(xmin, ymin, width, height));
          }
        }
      }
    }
  }
  else if (ConfigReader::getInstance()->haar_config.enable)
  {
    const cv::Size min_face_detect_size(50, 50);
    const cv::Size max_face_detect_size(450, 450);
    cascade.detectMultiScale(gray, faces,
        1.2, 3, 2
        ,min_face_detect_size, max_face_detect_size);
  }

  detect_time = 1000.0 * (clock()-time)/(double) CLOCKS_PER_SEC;
#if DEBUG
  printf( "Face Detection %ld faces and time is %g ms\n", faces.size(), detect_time);
#endif
  Rect temp;
  for(unsigned int i = 0; i < faces.size(); i++) {
    for(unsigned int j = i; j < faces.size(); j++) {
      if( faces[j].area() > faces[i].area() ) {
        temp = faces[j];
        faces[j] = faces[i];
        faces[i] = temp;
      }
    }
  }
  gray.release();
}

void FaceVerification::showFaceWindow(Mat& img, vector<Rect> faces)
{
  if (last_face_areas.size() != faces.size()) last_face_areas.clear();

  Rect temp;
  for(unsigned int i = 0; i < faces.size(); i++) {
    for(unsigned int j = i; j < faces.size(); j++) {
      if( faces[j].x < faces[i].x ) {
        temp = faces[j];
        faces[j] = faces[i];
        faces[i] = temp;
      }
    }
  }

  const int size = 300;
  if (faces.size() > 0) {
    Mat combine = Mat::zeros(size, size*faces.size(), img.type());
    for(unsigned int i = 0; i < faces.size(); i++) {
      const int x_center = faces[i].x + faces[i].width/2;
      const int y_center = faces[i].y + faces[i].height/2;
      int width = max(faces[i].width,350);
      int height = max(faces[i].height,350);
      int xmin = max(x_center - width/2, 0);
      int ymin = max(y_center - height/2, 0);
      if (xmin + width > img.cols)
        width = img.cols - xmin;
      if (ymin + height > img.rows)
        height = img.rows - ymin;

      Rect faceArea = Rect(xmin, ymin, width, height);

      bool needUpdate = true;
      for(unsigned int j = 0; j < last_face_areas.size(); j++) {
        if (getIoU(last_face_areas[j], faceArea) > 0.4){
          needUpdate = false;
          faceArea = last_face_areas[j];
          break;
        }
      }
      if (needUpdate) {
        last_face_areas.push_back(faceArea);
      }

      Mat face = img(faceArea);
      resize(face, face, Size(size, size));
      face.copyTo(combine(Rect(size*i, 0, size, size)));
      face.release();
    }
    imshow("Face Window", combine);
    combine.release();
  } else {
    destroyWindow("Face Window");
  }
}

void FaceVerification::faceAlignment(Mat& img, vector<Rect>& aligning_faces, vector<Point2f>& landmarks)
{
  try
  {
    unsigned int i;
    vector<Rect> faces = aligning_faces;
    vector<dlib::rectangle> dlibRectFaces;
    CVRect_to_DlibRect(dlibRectFaces, aligning_faces);
    Mat warped, final_face_mat;
    cv::Mat newimg(img.rows, img.cols, CV_8UC3);
    newimg = img.clone();
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    char outputImage[100];
    aligning_faces.clear();
    landmarks.clear();

    if (boost::filesystem::exists(CV_FACE_PREDICT_DIR)) {
      boost::filesystem::remove_all(CV_FACE_PREDICT_DIR);
    }
    boost::filesystem::create_directory(CV_FACE_PREDICT_DIR);

    clock_t time = clock();
    //#pragma omp parallel for num_threads(1)
    for (i = 0; i < dlibRectFaces.size(); ++i)
    {
      double left_eye_x;
      double left_eye_y;
      double right_eye_x;
      double right_eye_y;
      double left;
      double right;
      double bottom;
      if(ConfigReader::getInstance()->landmark_config.enable_caffe)
      {
        float* pose_detection;
        vector<Point2f> face_landmark = cfld_->Detect(img, faces[i], pose_detection);


        /*for(unsigned int j = 0; j < 3; j++) {
          LOG(INFO) << POSE_NAME[j] << " : " << pose_detection[j];
        }*/

        left_eye_x = (face_landmark[36].x + face_landmark[37].x + face_landmark[38].x + face_landmark[39].x + face_landmark[40].x + face_landmark[41].x)/6;
        left_eye_y = (face_landmark[36].y + face_landmark[37].y + face_landmark[38].y + face_landmark[39].y + face_landmark[40].y + face_landmark[41].y)/6;
        right_eye_x = (face_landmark[42].x + face_landmark[43].x + face_landmark[44].x + face_landmark[45].x + face_landmark[46].x + face_landmark[47].x)/6;
        right_eye_y = (face_landmark[42].y + face_landmark[43].y + face_landmark[44].y + face_landmark[45].y + face_landmark[46].y + face_landmark[47].y)/6;

        for (unsigned int j = 0; j < face_landmark.size(); j++) {
          landmarks.push_back(face_landmark[j]);
        }

        left =  face_landmark[1].x;
        right = face_landmark[15].x;
        bottom = face_landmark[8].y;
      }
      else
      {
        dlib::full_object_detection shape = pose_model(cimg, dlibRectFaces[i]);

        left_eye_x = (shape.part(36).x() + shape.part(37).x() + shape.part(38).x() + shape.part(39).x() + shape.part(40).x() + shape.part(41).x())/6;
        left_eye_y = (shape.part(36).y() + shape.part(37).y() + shape.part(38).y() + shape.part(39).y() + shape.part(40).y() + shape.part(41).y())/6;
        right_eye_x = (shape.part(42).x() + shape.part(43).x() + shape.part(44).x() + shape.part(45).x() + shape.part(46).x() + shape.part(47).x())/6;
        right_eye_y = (shape.part(42).y() + shape.part(43).y() + shape.part(44).y() + shape.part(45).y() + shape.part(46).y() + shape.part(47).y())/6;

        cv::Point2f nose;
        nose.x = shape.part(30).x();
        nose.y = shape.part(30).y();
        landmarks.push_back(nose);  //nose

        landmarks.push_back(Point2f((float)shape.part(48).x(),(float)shape.part(48).y()));  //right mouse
        landmarks.push_back(Point2f((float)shape.part(54).x(),(float)shape.part(54).y()));  //left mouse
        landmarks.push_back(Point2f((float)shape.part(0).x(),(float)shape.part(0).y()));    //right ear
        landmarks.push_back(Point2f((float)shape.part(16).x(),(float)shape.part(16).y()));  //left ear
        landmarks.push_back(Point2f((float)shape.part(8).x(),(float)shape.part(8).y()));  //bottom

        left =  shape.part(1).x();
        right = shape.part(15).x();
        bottom = shape.part(8).y();
      }

      double tan = (left_eye_y - right_eye_y )/(left_eye_x - right_eye_x );
      double rotate_angle = atan(tan) * 180 / M_PI;

      warped = Mat::zeros(newimg.rows, newimg.cols, newimg.type());
      Mat rot_mat = cv::getRotationMatrix2D(cv::Point(warped.cols/2.0f, warped.rows/2.0f), rotate_angle, 1.0);
      cv::warpAffine(newimg, warped, rot_mat, newimg.size());


      double width = right - left;
      left = left - width * 0.05;
      width = width * 1.1;

      double top = bottom - width;

      cv::Rect final_face;
      final_face.x = left;
      final_face.y = top;
      final_face.width = width;
      final_face.height = width;

      if (final_face.x < 0)
        final_face.x = 0;
      if (final_face.y < 0)
        final_face.y = 0;
      if (final_face.x + final_face.width > newimg.cols)
        final_face.width = newimg.cols - final_face.x;
      if (final_face.y + final_face.height > newimg.rows)
        final_face.height = newimg.rows - final_face.y;

      aligning_faces.push_back(final_face);
      final_face_mat = warped(final_face);

      sprintf(outputImage, "face_predict_%02d.jpg", i);
      cv::imwrite(CV_FACE_PREDICT_DIR + outputImage, final_face_mat);
      rot_mat.release();
    }
    alignment_time = 1000.0 * (clock()-time)/(double) CLOCKS_PER_SEC;
    int size = aligning_faces.size();
#if DEBUG
    printf( "Face Alignment %d faces and time is %g ms\n", size, alignment_time);
#endif
    newimg.release();
    warped.release();
    final_face_mat.release();
  }
  catch(cv::Exception& e)
  {
    printf("Face Alignment Caught cv::Exception: %s\n", e.what());
  }
  catch(dlib::error &err)
  {
    printf("Face Alignment Caught dlib::error: %s\n", err.what());
  }
  catch (...)
  {
    printf("Face Alignment Caught unknown exception\n");
  }
}

bool FaceVerification::faceVerification(int face_predict_num, vector<string>& face_ids, vector<cv::Rect>& faces) {
  bool no_strangers = true;
  double start,end;
  const float threshold = ConfigReader::getInstance()->sc_config.confidence_threshold;

  if (face_predict_num <= 0) {
     return false;
  }

  if (boost::filesystem::exists(CV_TEMP_DIR)) {
    boost::filesystem::remove_all(CV_TEMP_DIR);
  }
  boost::filesystem::create_directory(CV_TEMP_DIR);

  start = omp_get_wtime();
  int stranger_index = 0;
  for (int i = 0; i < face_predict_num; ++i)
  {
    char file_name[100];
    sprintf(file_name, "face_predict_%02d.jpg", i);
    string img_path = CV_FACE_PREDICT_DIR+file_name;
    if (!boost::filesystem::exists(img_path)) {
        printf("Can't find %s\n", img_path.c_str());
        continue;
    }

    fv_result = cfv_->verify_face(img_path, face_register_features, threshold);
    string face_id = "";
    bool intersects = ((leftRect & faces[i]).area() > 0) || ((rightRect & faces[i]).area() > 0);
    if (fv_result->index != -1) {
        face_id = face_register_paths[fv_result->index].filename().string();
        if (enable_auto_face_registration_retry) {
            if (fv_result->confidence > threshold
                    && !checkBlurryImage(img_path, intersects ? 60 : 150)) {
                faceRegistration(img_path, face_register_paths[fv_result->index].string());
            }
        }
    } else {
        fv_result = cfv_->verify_face(img_path, retry_face_register_features, threshold);
        if (fv_result->index != -1) {
            face_id = face_register_paths[fv_result->index].filename().string();
        } else if(!checkBlurryImage(img_path, intersects ? 60 : 150)) {
            no_strangers = false;
            sprintf(file_name, "face_stranger_%02d.jpg", stranger_index);
            string stranger_img_path = CV_TEMP_DIR+file_name;
            boost::filesystem::copy_file(img_path, stranger_img_path);
            stranger_index++;
        }
    }
    face_ids.push_back(face_id);
  }

  end = omp_get_wtime();
  predit_time = 1000.0 * (end-start);
#if DEBUG
  printf( "Face Verification %d faces and time is %f ms\n", face_predict_num, predit_time);
#endif
   return no_strangers;
}

static const Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)};

void drawLabel(Mat& img, Rect box, string label, Scalar color) {
  double fontScale = (1.7 * box.width) / 400;
  int id_y = box.y - (fontScale * 30);
  if (id_y < 0) id_y = 0;
  int rect_width = box.width * 0.5;
  if (label.find("Too Blurry!") != std::string::npos) {
      rect_width = box.width * 0.8;
      color = Scalar(0, 0, 255);
  }
  rectangle( img, Point(round(box.x), round(id_y)),
        Point(round(box.x + rect_width - 1), round(id_y + (fontScale * 30))),
        color,  CV_FILLED , 8, 0);
  cv::putText(img, label, Point(round(box.x + (fontScale * 5)), round(id_y + (fontScale * 25))),
        cv::FONT_HERSHEY_DUPLEX, fontScale, Scalar(255, 255, 255),
        2);
}

void FaceVerification::drawFaceBoxes(Mat& img, vector<Rect>& faces, vector<string>& face_ids)
{
#if DEBUG
  printf( "Draw Face Detection box on face\n");
#endif
  int i = 0;
  for(vector<Rect>::const_iterator box = faces.begin(); box != faces.end(); box++, i++ )
  {
    Scalar color = colors[i%8];
    rectangle( img, box->tl(), box->br(), color, 6, 8, 0);
    if (face_ids[i] != "" && face_ids[i].find("User#") == std::string::npos) {
        drawLabel(img, faces[i], face_ids[i], color);
    }
  }
  /*rectangle( img, leftRect.tl(), leftRect.br(), Scalar(255, 255, 255), 6, 8, 0);
  rectangle( img, rightRect.tl(), rightRect.br(), Scalar(255, 255, 255), 6, 8, 0);*/
}

void FaceVerification::drawFaceLandmarks(Mat& img, vector<Point2f>& landmarks)
{
    //Draw landmark on face
    for(unsigned int i = 0; i < landmarks.size(); i++)
    {
        //printf("dlibLandmark - point : %f , %f",imagePoints[a].x,imagePoints[a].y);
        Scalar color = CV_RGB(0,255,255);
        circle(img, landmarks[i], 1.3,  color, 3, 8, 0);
    }
}

void FaceVerification::drawDebugInformation(Mat& img, double fps) {
    char fpsMsg[100];
    Scalar color =  CV_RGB(0,0,255);
    sprintf(fpsMsg, "%.1fFPS", fps);
    cv::putText(img, fpsMsg, Point(5, 30), cv::FONT_HERSHEY_DUPLEX, 1.1, color, 2);
}

// implement Cv virtual fuction
void FaceVerification::reset(void)
{
  printf("reset!\n");
  if (init_libs_state_) loadRegisteredFaces();
}

int FaceVerification::detect(cv::Mat &img,
    vector<cv::Rect>& faces,
    vector<string>& face_ids,
    vector<Point2f>& landmarks)
{
  faces.clear();
  landmarks.clear();

  bool use_face_tracking = false;
  bool next_process = true;
  if (!init_libs_state_) {
    printf("Please initial Face CV Libs before face detectation!\n");
    return -1;
  }

  string face_predict_dir = CV_ROOT_DIR + "/face_predict/";
  if (boost::filesystem::exists(CV_FACE_PREDICT_DIR)) {
    boost::filesystem::remove_all(CV_FACE_PREDICT_DIR);
  }
  boost::filesystem::create_directory(CV_FACE_PREDICT_DIR);

  double start,end;
  start = omp_get_wtime();
  faceDetection(img, faces);
  if (faces.size() == 0) {
    face_detection_failed_count++;
    if (face_detection_failed_count > 5) {
        last_faces.clear();
        face_detection_failed_count = 0;
        stranger_count = 0;
        printf("Could not detect faces in predict frame!\n");
        next_process = false;
        face_ids.clear();
    } else if (last_faces.size() > 0) {
      faces = last_faces;
    }
  } else {
    if ((last_faces.size() == faces.size()) && (face_ids.size() == faces.size())) {
      use_face_tracking = true;
      for(unsigned int i = 0; i < last_faces.size(); i++) {
        if (getIoU(last_faces[i], faces[i]) == 0 || face_ids[i] == "") {
          use_face_tracking = false;
          break;
        }
      }
    }
    last_faces = faces;
    face_detection_failed_count = 0;
  }

  vector<Rect> aligning_faces(faces);
  leftRect = Rect(0, 0, img.cols * 0.2, img.rows);
  rightRect = Rect(img.cols * 0.8, 0, img.cols * 0.2, img.rows);
  if (next_process && !use_face_tracking) {
    faceAlignment(img, aligning_faces, landmarks);
    if ((landmarks.size() == 0 || aligning_faces.size() == 0)) {
      stranger_count = 0;
      printf("Could not find any face landmarks in predict frame\n");
      next_process = false;
    }
  }

  if (next_process) {
    if (!use_face_tracking) {
      face_ids.clear();
      if (!faceVerification(aligning_faces.size(), face_ids, faces)) {
        if (enable_auto_face_registration) {
          stranger_count++;
          const int max_stranger_count = ConfigReader::getInstance()->haar_config.enable ? 10 : 60;
          if (stranger_count > max_stranger_count) {
            std::vector<dlib::file> face_stranger_files = dlib::get_files_in_directory_tree(CV_TEMP_DIR,
                    dlib::match_endings(".png .PNG .jpeg .JPEG .jpg .JPG"), 10);
            for (unsigned int i = 0; i < face_stranger_files.size(); ++i)
            {
              if(faceRegistration(face_stranger_files[i].full_name()))
              {
                stranger_count = 0;
              }
            }
          }
        }
      } else {
        stranger_count = 0;
      }
    } else {
      printf("We use last result!\n");
      stranger_count = 0;
    }
  } else {
    face_ids.clear();
  }

  end = omp_get_wtime();
#if DEBUG
  printf( "Total time is %f ms\n", 1000.0 * (end-start));
#endif
  return 0;
}

bool FaceVerification::checkFaces(string img_path)
{
  cv::Mat img = imread(img_path);
  IplImage *face_img = new IplImage(img);
  const float sub_threshold = ConfigReader::getInstance()->yolo_config.sub_confidence_threshold;
  face_boxes = sub_yolo_detect(face_img, sub_threshold, .5, 1.2);
  if (face_boxes != NULL) {
    int detection_num = sub_yolo_get_detection_num();
    for (int i = 0; i < detection_num; ++i) {
      if (face_boxes[i].width > 0 && face_boxes[i].height > 0) {
        return true;
      }
    }
  }
  img.release();
  delete face_img;
  return false;
}

bool FaceVerification::checkBlurryImage(string img_path, int blur_threshold)
{
  cv::Mat grayImage = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
  int height = grayImage.rows * 0.4;
  grayImage = grayImage(Rect(0, grayImage.rows - height, grayImage.cols, height));
  cv::Mat laplacianImage;
  cv::Laplacian(grayImage, laplacianImage, CV_64F);
  cv::Scalar mu, sigma;
  cv::meanStdDev(laplacianImage, mu, sigma);
  double focusMeasure = sigma.val[0] * sigma.val[0];

  printf(" %s ------------ %lf \n", img_path.c_str(), focusMeasure);
  grayImage.release();
  laplacianImage.release();
  return (focusMeasure < blur_threshold);
}

bool FaceVerification::faceRegistration(string face_register_path, string face_id_dir)
{
  if (boost::filesystem::exists(face_register_path)) {
    if (!checkFaces(face_register_path)) {
      printf("No face for registration!\n");
      boost::filesystem::remove(face_register_path);
      return false;
    }

    bool is_retry = true;
    if (face_id_dir == "") {
        int dir_cnt = std::count_if(
                directory_iterator(CV_FACE_REGISTER_DIR),
                directory_iterator(),
                static_cast<bool(*)(const path&)>(is_directory));
        int face_id = dir_cnt + 1;
        is_retry = false;
        char face_dir_name[100];
        sprintf(face_dir_name, "%02d", face_id);
        face_id_dir = CV_FACE_REGISTER_DIR + boost::str(boost::format("User#%s") % face_dir_name);
        boost::filesystem::create_directory(face_id_dir);
    }

    std::vector<dlib::file> face_register_files = dlib::get_files_in_directory_tree(face_id_dir,
          dlib::match_endings(".png .PNG .jpeg .JPEG .jpg .JPG"), 100);
    int image_index = face_register_files.size() + 1;
    if (image_index >2) image_index = 2;
    char image_name[100];
    sprintf(image_name, "%d", image_index);
    string des_path = face_id_dir + boost::str(boost::format("/%s.jpg") % image_name);
    boost::filesystem::copy_file(face_register_path, des_path, copy_option::overwrite_if_exists);
    if (!is_retry) {
        boost::filesystem::path path(face_id_dir);
        face_register_paths.push_back(path);
        float* face_register_feature = cfv_->extract_feature(des_path);
        if(face_register_feature != NULL) {
            face_register_features.push_back(face_register_feature);
        }
    }
  } else {
    return false;
  }

  return true;
}

vector<boost::filesystem::path> FaceVerification::getFaceRegisterPaths() {
    return face_register_paths;
}
