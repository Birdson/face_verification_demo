#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <thread>
#include <vector>

#include "cv/face_verification.h"

using namespace std;
using namespace cv;

//#define ENABLE_MULTI_THREAD

class WebCamCap {
public:
    void captureFrame();
    void processFrame();
    static void CaptureThread(WebCamCap * cap);
    static void ProcessThread(WebCamCap * cap);
    bool keepRunning = true; // Will be changed by the external program.
    std::thread captureThread;
    std::thread processThread;
private:
    double total_time;
    double frame_count = 0, fps = 0;
    FaceVerification *fv = NULL;
    bool is_processing_frame = false;
    std::vector<cv::Mat> frame_buffer;
    cv::VideoCapture cap;
    vector<Rect> faces;
    vector<string> face_ids;
    vector<Point2f> landmarks;
    WebCamCap() { }
    static WebCamCap * s_instance;
public:
    static WebCamCap *instance();
};
