#include <string>
#include "cv/caffe_face_verification.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include <math.h> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <omp.h>

using std::string;
using std::static_pointer_cast;
using std::clock;
using std::clock_t;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::shared_ptr;
using caffe::vector;
using caffe::MemoryDataLayer;

namespace caffe {

static const string CV_ROOT_DIR = "./";
static const string CV_TEMP_FACE_IMAGE = CV_ROOT_DIR+".face.jpg";

template <typename T>
vector<size_t> ordered(vector<T> const& values) {
	vector<size_t> indices(values.size());
	std::iota(begin(indices), end(indices), static_cast<size_t>(0));

	std::sort(
		begin(indices), end(indices),
		[&](size_t a, size_t b) { return values[a] > values[b]; }
	);
	return indices;
}

static float GetCosineSimilarity(const float* V1,const float* V2, int len)
{
    double sim = 0.0d;
    int N = 0;
    N = len;
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

CaffeFaceVerification::CaffeFaceVerification(const string model_path,
                   const string weights_path) {
    CHECK_GT(model_path.size(), 0) << "Need a model definition to score.";
    CHECK_GT(weights_path.size(), 0) << "Need model weights to score.";

#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);
#endif

    caffe_net = new Net<float>(model_path, caffe::TEST);
    caffe_net->CopyTrainedLayersFrom(weights_path);

    caffe_net = new Net<float>(model_path, caffe::TEST);
    caffe_net->CopyTrainedLayersFrom(weights_path);
    Blob<float>* input_layer = caffe_net->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    string mean="0.0";
    scale_=1.0;
    if (caffe_net->name()=="sc50_net") {
        caffe_predit_size=110;
        caffe_blob_name="eltwise6";
    } else if (caffe_net->name()=="sc70_net") {
        mean="127.5";
        scale_=0.0078125;
        caffe_predit_size=57;
        caffe_blob_name="fc61";
    } else if (caffe_net->name()=="sc76_net") {
        scale_=0.00390625;
        caffe_predit_size=58;
        caffe_blob_name="fc777";
    } else if (caffe_net->name()=="sc79_net") {
        mean="127.5";
        scale_=0.0078125;
        caffe_predit_size=106;
        caffe_blob_name="fc1";
    }
    const string& mean_value=mean;

    /* Load the binaryproto mean file. */
    SetMean("", mean_value);
}

CaffeFaceVerification::~CaffeFaceVerification() {
    delete caffe_net;
}

/* Load the mean file in binaryproto format. */
void CaffeFaceVerification::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
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
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CaffeFaceVerification::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = caffe_net->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void CaffeFaceVerification::Preprocess(const cv::Mat& img,
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

  sample_normalized = sample_normalized * scale_;

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == caffe_net->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

float* CaffeFaceVerification::extract_feature(string img_path) {
    CHECK(caffe_net != NULL);

    if (caffe_predit_size == 57) {
        cv::Mat cv_img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
        cv::resize(cv_img, cv_img, cv::Size(60, 60));
        cv::Rect final_face;
        final_face.x = 1.5;
        final_face.y = 1.5;
        final_face.width = 58.5;
        final_face.height = 58.5;
        cv_img = cv_img(final_face);
        img_path = CV_TEMP_FACE_IMAGE;
        cv::imwrite(img_path, cv_img);
    } else if (caffe_predit_size == 58) {
        cv::Mat cv_img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
        cv::resize(cv_img, cv_img, cv::Size(60, 60));
        cv::Rect final_face;
        final_face.x = 1;
        final_face.y = 1;
        final_face.width = 59;
        final_face.height = 59;
        cv_img = cv_img(final_face);
        img_path = CV_TEMP_FACE_IMAGE;
        cv::imwrite(img_path, cv_img);
    } else if (caffe_predit_size == 106) {
        cv::Mat cv_img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
        cv::resize(cv_img, cv_img, cv::Size(110, 110));
        cv::Rect final_face;
        final_face.x = 2;
        final_face.y = 2;
        final_face.width = 108;
        final_face.height = 108;
        cv_img = cv_img(final_face);
        img_path = CV_TEMP_FACE_IMAGE;
        cv::imwrite(img_path, cv_img);
    }

    //shadow invariance
    cv::Mat cv_img = cv::imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
    for(int i = 0; i < cv_img.rows; i++){
        for(int j=0; j < cv_img.cols; j++){
            cv_img.at<uchar>(i,j) = std::sqrt( cv_img.at<uchar>(i,j) ) * 255.0/16.0;
       }
    }

    Blob<float>* input_layer = caffe_net->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    caffe_net->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(cv_img, &input_channels);


    float loss;
    vector<Blob<float>* > dummy_bottom_vec;
    double start = omp_get_wtime();
    const vector<Blob<float>*>& result = caffe_net->Forward(dummy_bottom_vec, &loss);
    double end = omp_get_wtime();
    double t_load = 1000.0 * (end-start);
    LOG(INFO) << "Elapsed time for extract feature time: " << t_load << " ms.";

    const shared_ptr<Blob<float> > feature_blob = caffe_net->blob_by_name(caffe_blob_name);
    int batch_size = feature_blob->num();
    dim_feature = feature_blob->count();

    LOG(INFO) << "dim_feature: " << dim_feature << " batch_size: " << batch_size;
    const float* data;

    data = feature_blob->cpu_data();

    if (dim_feature == 0) return NULL;

    int dim = dim_feature / batch_size;
    LOG(INFO) << "dim: " << dim << " batch_size: " << batch_size;
    float * feature = new float[dim];
    for(int i= 0; i < batch_size; i++) {
        for(int j= 0; j < dim; j++) {
            feature[j] = data[i*dim + j];
        }
    }
    return feature;
}

FaceVerificationData* CaffeFaceVerification::verify_face(string img_path, std::vector<float*>& face_register_features, const float threshold){
    FaceVerificationData* result = new FaceVerificationData;
    result->index = -1;
    float * feature = extract_feature(img_path);

    if (feature == NULL) return result;

    float max_sim = -1.0f;
    for (int i = 0; i < face_register_features.size(); ++i)
    {
        if (face_register_features[i] == NULL) continue;

        const float cos_sim = GetCosineSimilarity(face_register_features[i], feature, dim_feature);
        const float sim = 1.0f - (1.0f - cos_sim);
        LOG(INFO) << "face verification confidence: " << sim;
        if (sim >= threshold && sim > max_sim) {
            max_sim = sim;
            result->index = i;
            result->confidence = max_sim;
        }
    }

    free(feature);

    return result;
}

} // namespace caffe
