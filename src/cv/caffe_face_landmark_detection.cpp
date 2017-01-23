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

#include "caffe_face_landmark_detection.hpp"

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)

static const float M_left = -0.15;
static const float M_right = 1.15;
static const float M_top = -0.10;
static const float M_bottom = 1.25;

CaffeFaceLandmarkDetection::CaffeFaceLandmarkDetection(const string& model_file,
                   const string& weights_file,
                   const string& mean_file) {

#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);
}

CaffeFaceLandmarkDetection::~CaffeFaceLandmarkDetection() {
}

vector<cv::Point2f> CaffeFaceLandmarkDetection::Detect(const cv::Mat& img, cv::Rect& face, float* &pose_detection) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img(face), &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  const shared_ptr<Blob<float> > result_blob = net_->blob_by_name("68point");
  const float* result = result_blob->cpu_data();
  int batch_size = result_blob->num();
  int dim_result = result_blob->count();

  int dim = dim_result / batch_size;
  LOG(INFO) << "dim: " << dim;
  float * detection = new float[dim];
  for(int i= 0; i < batch_size; i++) {
    for(int j= 0; j < dim; j++) {
      detection[j] = result[i*dim + j] * 224/2 + 224/2;
      //LOG(INFO) << "detection[" << j << "] : " << detection[j];
    }
  }

  float cut_size_left = face.x + M_left * face.width;
  float cut_size_right = (face.x + face.width) + (M_right - 1) * face.width;
  float cut_size_top = face.y + M_top * face.height;
  float cut_size_bottom = (face.y + face.height) + (M_bottom - 1) * face.height;

  float scale_x = (cut_size_right - cut_size_left)*0.75/224;
  float scale_y = (cut_size_bottom - cut_size_top)*0.75/224;

  vector<cv::Point2f> face_landmarks;
  for(int i = 0; i < dim; i++) {
    if (i % 2 == 0) {
      float x = float(detection[i] * scale_x + face.x);
      float y = float(detection[i+1] * scale_y + face.y);
      face_landmarks.push_back(cv::Point2f(x, y));
    }
  }


  const shared_ptr<Blob<float> > pose_result_blob = net_->blob_by_name("poselayer");
  const float* pose_result = pose_result_blob->cpu_data();
  batch_size = pose_result_blob->num();
  dim_result = pose_result_blob->count();

  dim = dim_result / batch_size;
  LOG(INFO) << "dim: " << dim;
  pose_detection = new float[dim];
  for(int i= 0; i < batch_size; i++) {
    for(int j= 0; j < dim; j++) {
      pose_detection[j] = pose_result[i*dim + j] * 50;
    }
  }
  
  return face_landmarks;
}

/* Load the mean file in binaryproto format. */
void CaffeFaceLandmarkDetection::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CaffeFaceLandmarkDetection::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void CaffeFaceLandmarkDetection::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
