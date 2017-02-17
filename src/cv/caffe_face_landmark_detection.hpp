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

#ifndef CAFFE_LANDMARK_DETECTION_HPP_
#define CAFFE_LANDMARK_DETECTION_HPP_

#include <string>
#include "caffe/caffe.hpp"
#include <opencv2/opencv.hpp>

using std::string;

namespace caffe {

class CaffeFaceLandmarkDetection {
  public:
    CaffeFaceLandmarkDetection(const string& model_file,
           const string& weights_file,
           const string& mean_file);
    ~CaffeFaceLandmarkDetection();
    vector<cv::Point2f> Detect(const cv::Mat& img, cv::Rect& face, float* &pose_detection);

  private:
    void SetMean(const string& mean_file);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

} // namespace caffe

#endif
