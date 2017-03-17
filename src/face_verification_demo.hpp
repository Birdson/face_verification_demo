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
    void startVideoCapture();
    void processFrame();
    static void ProcessThread(WebCamCap * cap);
    bool keepRunning = true; // Will be changed by the external program.
    std::thread captureThread;
    std::thread processThread;
private:
    double total_time;
    double frame_count = 0, fps = 0;
    FaceVerification *fv = NULL;
    bool is_start_processing_frame = false;
    std::vector<cv::Mat> frame_buffer;
    cv::VideoCapture cap;
    vector<FaceData> face_datas;
    WebCamCap() { }
    static WebCamCap * s_instance;
    pthread_t detectThread;
public:
    static WebCamCap *instance();
};
