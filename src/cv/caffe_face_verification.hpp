#ifndef CAFFE_FACE_VERIFICATION_HPP_
#define CAFFE_FACE_VERIFICATION_HPP_

#include <string>
#include <opencv2/core/core.hpp>
#include "caffe/caffe.hpp"

using std::string;

namespace caffe {

class CaffeFaceVerification
{
public:
    CaffeFaceVerification(const string model_path,
                   const string weights_path);
    ~CaffeFaceVerification();
    float* extract_feature(const string img_path);
    int verify_face(string img_path, std::vector<float*>& face_register_features, const float threshold);

private:
    Net<float> *caffe_net;
    int num_channels_;
    cv::Size input_geometry_;
    cv::Mat mean_;
    float scale_;
    int caffe_predit_size;
    string caffe_blob_name;
    int dim_feature;
    void SetMean(const string& mean_file, const string& mean_value);
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
};

} // namespace caffe

#endif
