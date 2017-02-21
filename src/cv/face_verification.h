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

#ifndef _FACE_VERIFICATION_H_
#define _FACE_VERIFICATION_H_

#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include "caffe_face_detection.hpp"
#include "caffe_face_landmark_detection.hpp"
#include "caffe_face_verification.hpp"

#include <boost/filesystem.hpp>

#include "yolo_detector.h"

using namespace std;

typedef struct {
  cv::Rect face_box;
  cv::Point2f face_landmark;
  std::string face_id;
} Face;

class FaceVerification
{
  public :
    FaceVerification();
    ~FaceVerification();

    //Algorithms Part
    /*===========================================================================
     * FUNCTION  : faceRegistration
     *
     * DESCRIPTION  : Loading pre setting face taget
     *
     * PARAMETERS  :
     *   @face_register_path : new face image path for face registeration
     *   @face_id_dir : assign pre registered face folder 
     *
     * RETURN  : success or not 
     *==========================================================================*/
    bool faceRegistration(string face_register_path, string face_id_dir="");

    /*===========================================================================
     * FUNCTION  : reset
     *
     * DESCRIPTION  : Reseting to ready state
     *==========================================================================*/
    void reset(void);

    /*===========================================================================
     * FUNCTION  : detect
     *
     * DESCRIPTION  : Detect and verify face in image
     *
     * PARAMETERS  :
     *   @img : source image refrence
     *   @faces : detected face region
     *   @face_ids : face verification result
     *   @landmarks : face landmark result
     *
     * RETURN  : status
     *==========================================================================*/
    int detect(cv::Mat &img, vector<cv::Rect>& faces, vector<string>& face_ids, vector<cv::Point2f>& landmarks);

    /*===========================================================================
     * FUNCTION  : faceDetection
     *
     * DESCRIPTION  : Detect face in image
     *
     * PARAMETERS  :
     *   @img : source image refrence
     *   @faces : detected face region
     *
     *==========================================================================*/
    void faceDetection(cv::Mat& img, vector<cv::Rect>& faces);

    /*===========================================================================
     * FUNCTION  : faceAlignment
     *
     * DESCRIPTION  : Align face for detected faces
     *
     * PARAMETERS  : 
     *   @img : source image refrence
     *   @aligning_faces : face region after alignment
     *   @landmarks : face landmarks
     *
     *==========================================================================*/
    void faceAlignment(cv::Mat& img, vector<cv::Rect>& aligning_faces, vector<cv::Point2f>& landmarks);

    /*===========================================================================
     * FUNCTION  : faceVerification
     *
     * DESCRIPTION  : Verify face for aligned faces
     *
     * PARAMETERS  :
     *   @face_predict_num : the number of face for verification
     *   @face_ids : face verification result
     *   @faces : detected face region
     * 
     * RETURN  : success or not
     *==========================================================================*/
    bool faceVerification(int face_predict_num, vector<string>& face_ids, vector<cv::Rect>& faces);

    /*===========================================================================
     * FUNCTION  : getFaceRegisterPaths
     *
     * DESCRIPTION  : Get current registered face image paths
     * 
     * RETURN  : face image paths
     *==========================================================================*/
    vector<boost::filesystem::path> getFaceRegisterPaths();

    //UI Part
    void showFaceWindow(cv::Mat& img, cv::Mat& combine, std::vector<cv::Rect> faces);

    bool enable_face_registration;
    bool enable_face_registration_retry;

  private :
    bool init_libs_state_;
    cv::CascadeClassifier cascade;
    dlib::shape_predictor pose_model;
    caffe::CaffeFaceDetection *cfd_;
    caffe::CaffeFaceLandmarkDetection *cfld_;
    caffe::CaffeFaceVerification *cfv_;
    std::string face_registration_dir;
    std::vector<boost::filesystem::path> face_register_paths;
    std::vector<float*> face_register_features;
    std::vector<float*> retry_face_register_features;
    cv::Rect leftRect, rightRect;
    int stranger_count;
    int face_detection_failed_count;
    std::vector<cv::Rect> last_faces;
    std::vector<cv::Rect> last_face_areas;
    FaceVerificationData* fv_result;
    face *face_boxes;

    bool init(void);
    bool initFaceDetection(void);
    bool initFaceLandmarkDetection(void);
    bool initFaceVerification(void);
    bool checkFaces(string img_path);
    bool checkBlurryImage(string img_path, int blur_threshold=130);
    void loadRegisteredFaces(void);
};

#endif /* _FACE_VERIFICATION_H_ */
