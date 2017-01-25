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

#include "cv/face_verification.h"

#include <string>

#include "camera_cv_state.h"

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
#define DEBUG_FACE_VERIFICATION 0 //For integration test if there is no registered face

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace boost::lambda;

static double total_time = 0, detect_time = 0, alignment_time = 0, predit_time = 0;
static double frame_count = 0, fps = 0;
static const cv::Size CAFFE_FACE_SIZE(200, 200);
static const float CAFFE_STRNGER_THD = 0.55;
static const bool ENABLE_AUTO_FACE_REGISTER = true;
static const std::string CV_ROOT_DIR = "./";
static const std::string CV_MODELS_DIR = CV_ROOT_DIR + "models/";
#ifdef ENABLE_CAFFE_FACE_DETECTION
static const cv::Size MIN_FACE_DETECT_SIZE(50, 50);
static const int MAX_STRANGER_COUNT = 60;
static const float CONFIDENCE_THRESHOLD = 0.4;
static const std::string FACE_DETECTION_MODEL_PATH = CV_MODELS_DIR + "pega_face_detection_deploy.prototxt";
static const std::string FACE_DETECTION_WEIGHT_PATH = CV_MODELS_DIR + "pega_face_detection.caffemodel";
#else
static const cv::Size MIN_FACE_DETECT_SIZE(50, 50);
static const cv::Size MAX_FACE_DETECT_SIZE(450, 450);
static const int MAX_STRANGER_COUNT = 10;
static const std::string FACE_DETECTION_MODEL_PATH = CV_MODELS_DIR + "haarcascade_frontalface_alt2.xml";
#endif
static const std::string FACE_LANDMARK_DETECTION_MODEL_PATH = CV_MODELS_DIR + "pega_68_face_landmarks.dat";
static const std::string CNN_FACE_LANDMARK_DETECTION_MODEL_PATH = CV_MODELS_DIR + "68point_with_pose_deploy.prototxt";
static const std::string CNN_FACE_LANDMARK_DETECTION_WEIGHT_PATH = CV_MODELS_DIR + "68point_with_pose.caffemodel";
static const std::string CNN_FACE_LANDMARK_DETECTION_MEAN_PATH = CV_MODELS_DIR + "VGG_mean.binaryproto";
static const std::string FACE_VERIFICATION_WEIGHT_PATH = CV_MODELS_DIR + "pega_face_verification.caffemodel";
static const std::string FACE_VERIFICATION_MODEL_PATH = CV_MODELS_DIR + "pega_face_verification_deploy.prototxt";
static const std::string CV_FACE_REGISTER_DIR = CV_ROOT_DIR + "face_register/";
static const std::string CV_FACE_PREDICT_DIR= CV_ROOT_DIR + "face_predict/";
static const std::string CV_TEMP_DIR = CV_ROOT_DIR + ".temp/";

static const std::string POSE_NAME[] = {"Pitch", "Yaw", "Roll"};

inline bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

inline void CVRect_to_DlibRect(vector<dlib::rectangle>& d_rect, vector<Rect>& cv_rect)
{
  dlib::rectangle rect;

  d_rect.clear();

  for(int i=0; i < cv_rect.size(); i++)
  {
    rect.set_left(cv_rect[i].x);
    rect.set_top(cv_rect[i].y);
    rect.set_right(cv_rect[i].x + cv_rect[i].height); //x + height
    rect.set_bottom(cv_rect[i].y + cv_rect[i].width); // y + width
    d_rect.push_back(rect);
  }
}

static float GetCosineSimilarity(const float* V1,const float* V2)
{
    double sim = 0.0d;
    int N = sizeof(V1);
    printf( "N is %d", N);
    double dot = 0.0d;
    double mag1 = 0.0d;
    double mag2 = 0.0d;
    for (int n = 0; n < N; n++)
    {
        dot += V1[n] * V2[n];
        mag1 += pow(V1[n], 2);
        mag2 += pow(V2[n], 2);
    }

    return dot / (sqrt(mag1) * sqrt(mag2));
}

FaceVerification::FaceVerification()
{
  cfv_ = NULL;
  if (!boost::filesystem::exists(CV_TEMP_DIR)) {
    boost::filesystem::create_directory(CV_TEMP_DIR);
  }
  init_libs_state_ = init();
  if (init_libs_state_) loadFaceFeatures();
}

FaceVerification::~FaceVerification()
{
  delete cfv_;
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

void FaceVerification::loadFaceFeatures(void)
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
  if (!file_exists(FACE_DETECTION_MODEL_PATH))  {
    CHECK_GT(FACE_DETECTION_MODEL_PATH.size(), 0) << "Need a Face Detection Model to detect face.";
    printf("Can't find Face Detection Model %s\n", FACE_DETECTION_MODEL_PATH.c_str());
    return false;
  }

#ifdef ENABLE_CAFFE_FACE_DETECTION
  if (!file_exists(FACE_DETECTION_WEIGHT_PATH))  {
    CHECK_GT(FACE_DETECTION_WEIGHT_PATH.size(), 0) << "Need a Face Detection Weight to detect face.";
    printf("Can't find Face Detection Weight %s\n", FACE_DETECTION_WEIGHT_PATH.c_str());
    return false;
  }
#endif

#if DEBUG
  printf("Initial Face Detection: %s\n", FACE_DETECTION_MODEL_PATH.c_str());
#endif
  bool init_state = false;
  try
  {
    clock_t start, end;
    double costtime;
    start = clock();

#ifdef ENABLE_CAFFE_FACE_DETECTION
    const string& mean_value = "104,117,123";
    cfd_ = new caffe::CaffeFaceDetection(FACE_DETECTION_MODEL_PATH, FACE_DETECTION_WEIGHT_PATH, "", mean_value);
    init_state = true;
#else
    init_state = cascade.load(FACE_DETECTION_MODEL_PATH);
#endif

    end = clock();
    costtime = 1000.0 * (end-start)/(double) CLOCKS_PER_SEC;
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
  if (!file_exists(FACE_LANDMARK_DETECTION_MODEL_PATH))  {
    CHECK_GT(FACE_LANDMARK_DETECTION_MODEL_PATH.size(), 0) << "Need a dlib shape to landmark face.";
    printf("Can't find dlib shape %s\n", FACE_LANDMARK_DETECTION_MODEL_PATH.c_str());
  }

#if DEBUG
  printf("Initial Dlib Shape Predictor: %s\n", FACE_LANDMARK_DETECTION_MODEL_PATH.c_str());
#endif
  bool init_state = false;
  try
  {
    clock_t start, end;
    double costtime;
    start = clock();

#ifdef ENABLE_CAFFE_FACE_LANDMARK_DETECTION
    cfld_ = new caffe::CaffeFaceLandmarkDetection(CNN_FACE_LANDMARK_DETECTION_MODEL_PATH,
            CNN_FACE_LANDMARK_DETECTION_WEIGHT_PATH, CNN_FACE_LANDMARK_DETECTION_MEAN_PATH);
#else
    dlib::deserialize(FACE_LANDMARK_DETECTION_MODEL_PATH) >> pose_model;
#endif
    end = clock();
    costtime = 1000.0 * (end-start)/(double) CLOCKS_PER_SEC;
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
  if (!file_exists(FACE_VERIFICATION_MODEL_PATH))  {
    CHECK_GT(FACE_VERIFICATION_MODEL_PATH.size(), 0) << "Need a Model Definition File to verify face.";
    printf("Can't find Model Definition %s\n", FACE_VERIFICATION_MODEL_PATH.c_str());
    return false;
  }
  if (!file_exists(FACE_VERIFICATION_WEIGHT_PATH))  {
    CHECK_GT(FACE_VERIFICATION_WEIGHT_PATH.size(), 0) << "Need a Model Definition File to verify face.";
    printf("Can't find Model Definition %s\n", FACE_VERIFICATION_WEIGHT_PATH.c_str());
    return false;
  }
#if DEBUG
  printf("Initial Cafffe Model Definition: %s\n", FACE_VERIFICATION_MODEL_PATH.c_str());
  printf("Initial Cafffe Model Weights: %s\n", FACE_VERIFICATION_WEIGHT_PATH.c_str());
#endif
  clock_t start, end;
  double costtime;
  start = clock();
  cfv_ = new caffe::CaffeFaceVerification(FACE_VERIFICATION_MODEL_PATH, FACE_VERIFICATION_WEIGHT_PATH);
  end = clock();
  costtime = 1000.0 * (end-start)/(double) CLOCKS_PER_SEC;
#if DEBUG
  printf("Load Caffe Net time is %f ms\n", costtime);
#endif
  return true;
}

void FaceVerification::faceDetection(Mat& img, vector<Rect>& faces)
{
  Mat gray;
  cvtColor( img, gray, CV_BGR2GRAY );

  double start = (double)cvGetTickCount();
#ifdef ENABLE_CAFFE_FACE_DETECTION
  std::vector<std::vector<float> > detections = cfd_->Detect(img);
  const int detection_size = 7;
  for (int i = 0; i < detections.size(); ++i) {
        const std::vector <float> detection = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].

        CHECK_EQ(detection.size(), detection_size);

        const float score = detection[2];
        if (score >= CONFIDENCE_THRESHOLD) {
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

            if (min(width, height) > MIN_FACE_DETECT_SIZE.width) {
                const float ratio = ((float)min(width, height)/(float)max(width, height));
                if (ratio > 0.65) {
                    faces.push_back(Rect(xmin, ymin, width, height));
                }
            }
        }
    }
#else
  cascade.detectMultiScale(gray, faces,
        1.2, 3, 2
        ,MIN_FACE_DETECT_SIZE, MAX_FACE_DETECT_SIZE);
#endif

  double end = (double)cvGetTickCount();
  detect_time = (end - start)/((double)cvGetTickFrequency()*1000.);
  int size = faces.size();
#if DEBUG
  printf( "Face Detection %d faces and time is %g ms\n", size, detect_time);
#endif
  Rect temp;
  for( int i = 0; i < faces.size(); i++) {
    for( int j = i; j < faces.size(); j++) {
      if( faces[j].area() > faces[i].area() ) {
        temp = faces[j];
        faces[j] = faces[i];
        faces[i] = temp;
      }
    }
  }
  if (faces.size() > 0) {
    Mat face = img(faces[0]);
    resize(face, face, Size(300, 300));
    imshow("Face Detection", face);
  } else {
    destroyWindow("Face Detection");
  }
  gray.release();
}

void FaceVerification::faceAlignment(Mat& img, vector<Rect>& aligning_faces, vector<Point2f >& landmarks)
{
  try
  {
    int i;
    vector<Rect> faces = aligning_faces;
    clock_t start,end;
    vector<dlib::rectangle> dlibRectFaces;
    CVRect_to_DlibRect(dlibRectFaces, aligning_faces);
    Mat gray, warped, final_face_mat;
    cvtColor(img, gray, CV_BGR2GRAY);
    cv::Mat newimg(img.rows, img.cols, CV_8UC3);
    newimg = gray.clone();
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    char outputImage[100];
    aligning_faces.clear();
    landmarks.clear();

    if (boost::filesystem::exists(CV_FACE_PREDICT_DIR)) {
      boost::filesystem::remove_all(CV_FACE_PREDICT_DIR);
    }
    boost::filesystem::create_directory(CV_FACE_PREDICT_DIR);

    start = clock();
    //#pragma omp parallel for num_threads(1)
    for (i = 0; i < dlibRectFaces.size(); ++i)
    {
#ifdef ENABLE_CAFFE_FACE_LANDMARK_DETECTION
      float* pose_detection;
      vector<Point2f> face_landmark = cfld_->Detect(img, faces[i], pose_detection);


      /*for(int j = 0; j < 3; j++) {
        LOG(INFO) << POSE_NAME[j] << " : " << pose_detection[j];
      }*/

      double left_eye_x = (face_landmark[36].x + face_landmark[37].x + face_landmark[38].x + face_landmark[39].x + face_landmark[40].x + face_landmark[41].x)/6;
      double left_eye_y = (face_landmark[36].y + face_landmark[37].y + face_landmark[38].y + face_landmark[39].y + face_landmark[40].y + face_landmark[41].y)/6;
      double right_eye_x = (face_landmark[42].x + face_landmark[43].x + face_landmark[44].x + face_landmark[45].x + face_landmark[46].x + face_landmark[47].x)/6;
      double right_eye_y = (face_landmark[42].y + face_landmark[43].y + face_landmark[44].y + face_landmark[45].y + face_landmark[46].y + face_landmark[47].y)/6;

      for (int j = 0; j < face_landmark.size(); j++) {
        landmarks.push_back(face_landmark[j]);
      }

      double left =  face_landmark[1].x;
      double right = face_landmark[15].x;
      double bottom = face_landmark[8].y;

#else
      dlib::deserialize(FACE_LANDMARK_DETECTION_MODEL_PATH) >> pose_model;

      double left_eye_x = (shape.part(36).x() + shape.part(37).x() + shape.part(38).x() + shape.part(39).x() + shape.part(40).x() + shape.part(41).x())/6;
      double left_eye_y = (shape.part(36).y() + shape.part(37).y() + shape.part(38).y() + shape.part(39).y() + shape.part(40).y() + shape.part(41).y())/6;
      double right_eye_x = (shape.part(42).x() + shape.part(43).x() + shape.part(44).x() + shape.part(45).x() + shape.part(46).x() + shape.part(47).x())/6;
      double right_eye_y = (shape.part(42).y() + shape.part(43).y() + shape.part(44).y() + shape.part(45).y() + shape.part(46).y() + shape.part(47).y())/6;

      cv::Point2f nose;
      nose.x = shape.part(30).x();
      nose.y = shape.part(30).y();
      landmarks.push_back(nose);  //nose

      landmarks.push_back(Point2f((float)shape.part(48).x(),(float)shape.part(48).y()));  //right mouse
      landmarks.push_back(Point2f((float)shape.part(54).x(),(float)shape.part(54).y()));  //left mouse
      landmarks.push_back(Point2f((float)shape.part(0).x(),(float)shape.part(0).y()));    //right ear
      landmarks.push_back(Point2f((float)shape.part(16).x(),(float)shape.part(16).y()));  //left ear
      landmarks.push_back(Point2f((float)shape.part(8).x(),(float)shape.part(8).y()));  //bottom

      double left =  shape.part(1).x();
      double right = shape.part(15).x();
      double bottom = shape.part(8).y();
#endif

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

      //cv::resize(final_face_mat, final_face_mat, CAFFE_FACE_SIZE);
      sprintf(outputImage, "face_predict_%02d.jpg", i);
      cv::imwrite(CV_FACE_PREDICT_DIR + outputImage, final_face_mat);
      rot_mat.release();
    }
    end = clock();
    alignment_time = 1000.0 * (end-start)/(double) CLOCKS_PER_SEC;
    int size = aligning_faces.size();
#if DEBUG
    printf( "Face Alignment %d faces and time is %g ms\n", size, alignment_time);
#endif
    gray.release();
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
    if (!file_exists(img_path)) {
        printf("Can't find %s\n", img_path.c_str());
        continue;
    }

    bool intersects = ((leftRect & faces[i]).area() > 0) || ((rightRect & faces[i]).area() > 0);
    if (checkBlurryImage(img_path, intersects ? 60 : 150)) {
        //face_ids.push_back("Too Blurry!");
        //continue;
    }

    int index = cfv_->verify_face(img_path, face_register_features, CAFFE_STRNGER_THD);
    //string face_id = "Stranger";
    string face_id = "";
    if (index != -1) {
        face_id = face_register_paths[index].filename().string();
        if (ENABLE_AUTO_FACE_REGISTER) faceRegister(img_path, face_register_paths[index].string());
    } else {
        index = cfv_->verify_face(img_path, retry_face_register_features, CAFFE_STRNGER_THD);
        if (index != -1) {
            face_id = face_register_paths[index].filename().string();
            if (ENABLE_AUTO_FACE_REGISTER) faceRegister(img_path, face_register_paths[index].string());
        } else {
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
  if (label == "Stranger") {
    rect_width = box.width * 0.65;
    color = Scalar(0, 0, 255);
  } else if (label == "Too Blurry!") {
    rect_width = box.width * 0.8;
    color = Scalar(0, 0, 255);
  }
  rectangle( img, cvPoint(cvRound(box.x), cvRound(id_y)),
        cvPoint(cvRound(box.x + rect_width - 1), cvRound(id_y + (fontScale * 30))),
        color,  CV_FILLED , 8, 0);
  cv::putText(img, label, Point(cvRound(box.x + (fontScale * 5)), cvRound(id_y + (fontScale * 25))),
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
    rectangle( img, cvPoint(cvRound(box->x), cvRound(box->y)),
        cvPoint(cvRound(box->x + box->width-1), cvRound(box->y + box->height-1)),
        color, 6, 8, 0);
    if (face_ids[i] != "") drawLabel(img, faces[i], face_ids[i], color);
  }
  /*rectangle( img, cvPoint(cvRound(leftRect.x), cvRound(leftRect.y)),
        cvPoint(cvRound(leftRect.x + leftRect.width-1), cvRound(leftRect.y + leftRect.height-1)),
        Scalar(255, 255, 255), 6, 8, 0);
  rectangle( img, cvPoint(cvRound(rightRect.x), cvRound(rightRect.y)),
        cvPoint(cvRound(rightRect.x + rightRect.width-1), cvRound(rightRect.y + rightRect.height-1)),
        Scalar(255, 255, 255), 6, 8, 0);*/
}

void FaceVerification::drawFaceLandmarks(Mat& img, vector<Point2f >& landmarks)
{
    //Draw landmark on face
    for(int i = 0; i < landmarks.size(); i++)
    {
        //printf("dlibLandmark - point : %f , %f",imagePoints[a].x,imagePoints[a].y);
        Scalar color = CV_RGB(0,255,255);
        circle(img, landmarks[i], 1.3,  color, 3, 8, 0);
    }
}

void FaceVerification::drawDebugInformation(Mat& img) {
    char fpsMsg[100];
    Scalar color =  CV_RGB(0,0,255);
    total_time = total_time + detect_time + alignment_time + predit_time;
    if (frame_count == 30) {
        fps = 1000 / (total_time / frame_count);
        frame_count = 0;
        total_time = 0;
    }
    sprintf(fpsMsg, "%.1fFPS", fps);
    cv::putText(img, fpsMsg, Point(5, 30), cv::FONT_HERSHEY_DUPLEX, 1.1, color, 2);
}

// implement Cv virtual fuction
void FaceVerification::reset(void)
{
  printf("reset!\n");
  if (init_libs_state_) loadFaceFeatures();
}

int FaceVerification::detect(cv::Mat &img, vector<cv::Rect>& faces)
{
  bool next_process = true;
  frame_count++;
  if (!init_libs_state_) {
    printf("Please initial Face CV Libs before face detectation!\n");
    return CV_FR_INIT_FAIL;
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
    stranger_count = 0;
    printf("Could not detect faces in predict frame!\n");
    next_process = false;
  }

  vector<Point2f > landmarks;
  vector<Rect> aligning_faces(faces);
  faceAlignment(img, aligning_faces, landmarks);
  leftRect = Rect(0, 0, img.cols * 0.2, img.rows);
  rightRect = Rect(img.cols * 0.8, 0, img.cols * 0.2, img.rows);
  if (next_process && (landmarks.size() == 0 || aligning_faces.size() == 0)) {
    stranger_count = 0;
    printf("Could not find any face landmarks in predict frame\n");
    next_process = false;
  }

  vector<string> face_ids;
  if (next_process && !faceVerification(aligning_faces.size(), face_ids, faces)) {
    if (ENABLE_AUTO_FACE_REGISTER) {
        stranger_count++;
        if (stranger_count > MAX_STRANGER_COUNT) {
            stranger_count = 0;
            std::vector<dlib::file> face_stranger_files = dlib::get_files_in_directory_tree(CV_TEMP_DIR,
                    dlib::match_endings(".png .PNG .jpeg .JPEG .jpg .JPG"), 10);
            for (int i = 0; i < face_stranger_files.size(); ++i)
            {
                faceRegister(face_stranger_files[i].full_name());
            }
        }
    }
  } else {
    stranger_count = 0;
  }
  end = omp_get_wtime();
#if DEBUG
  printf( "Total time is %f ms\n", 1000.0 * (end-start));
#endif

#ifdef ENABLE_DRAW_FACE_BOXS
  if (faces.size() > 0)
    drawFaceBoxes(img, faces, face_ids);
#endif

#ifdef ENABLE_DRAW_FACE_LANDMARKS
  if (landmarks.size() > 0)
    drawFaceLandmarks(img, landmarks);
#endif

#ifdef ENABLE_DRAW_DEBUG_INFORMATION
  drawDebugInformation(img);
#endif

  return 0;
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

bool FaceVerification::faceRegister(string face_register_path, string face_id_dir)
{
  if (boost::filesystem::exists(face_register_path)) {
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
        face_id_dir = CV_FACE_REGISTER_DIR + boost::str(boost::format("User%s") % face_dir_name);
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
  }

  return true;
}


