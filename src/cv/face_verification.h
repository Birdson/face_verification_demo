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

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <dlib/image_processing.h>
#include "caffe_face_detection.hpp"
#include "caffe_face_landmark_detection.hpp"
#include "caffe_face_verification.hpp"

#include <boost/filesystem.hpp>

#define ENABLE_CAFFE_FACE_DETECTION
#define ENABLE_CAFFE_FACE_LANDMARK_DETECTION

#define ENABLE_DRAW_FACE_BOXS
#define ENABLE_DRAW_FACE_LANDMARKS
#define ENABLE_DRAW_DEBUG_INFORMATION

using namespace std;

class FaceVerification
{
  public : 
    /*===========================================================================
     * FUNCTION  : faceRegister
     *
     * DESCRIPTION  : Loading pre setting face taget
     *
     * PARAMETERS  : 
     *   @face : pre setting face path
     *
     * RETURN  : success or not 
     *==========================================================================*/
    bool faceRegister(string face_register_path, string face_id_dir="");

    /*===========================================================================
     * FUNCTION  : reset
     *
     * DESCRIPTION  : Reseting to ready state
     *==========================================================================*/
    void reset(void);

    /*===========================================================================
     * FUNCTION  : detect
     *
     * DESCRIPTION  : Detect taget in image
     *
     * PARAMETERS  : 
     *   @img : source image refrence
     *   @roi : detected target region
     *
     * RETURN  : 
     *==========================================================================*/
    int detect(cv::Mat &img, vector<cv::Rect>& faces);

    void faceDetection(cv::Mat& img, vector<cv::Rect>& faces);
    void faceAlignment(cv::Mat& img, vector<cv::Rect>& aligning_faces, vector<cv::Point2f >& landmarks);
    bool faceVerification(int face_predict_num, vector<string>& face_ids, vector<cv::Rect>& faces);

    FaceVerification();
    ~FaceVerification();

  private :
    bool init_libs_state_;
    cv::CascadeClassifier cascade;
    dlib::shape_predictor pose_model;
    caffe::CaffeFaceDetection *cfd_;
    caffe::CaffeFaceLandmarkDetection *cfld_;
    caffe::CaffeFaceVerification *cfv_;
    std::vector<boost::filesystem::path> face_register_paths;
    std::vector<float*> face_register_features;
    std::vector<float*> retry_face_register_features;
    cv::Rect leftRect, rightRect;
    bool init(void);
    bool initFaceDetection(void);
    bool initFaceLandmarkDetection(void);
    bool initFaceVerification(void);
    bool checkBlurryImage(string img_path, int blur_threshold=130);
    void loadFaceFeatures(void);
    void drawFaceBoxes(cv::Mat& img, vector<cv::Rect>& faces, vector<string >& face_ids);
    void drawFaceLandmarks(cv::Mat& img, vector<cv::Point2f >& landmarks);
    void drawDebugInformation(cv::Mat& img);
    int stranger_count;
};

#endif /* _FACE_VERIFICATION_H_ */
