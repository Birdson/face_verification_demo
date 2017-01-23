#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>

#include "cv/face_verification.h"
#include "camera_cv_state.h"
#include "cv/caffe_face_verification.hpp"

using namespace cv;

int main(int argc, char **argv){
    clock_t start;
    double total_detect_sec;
    Mat frame;
    Mat output_frame;
    VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())  // check if succeeded to connect to the camera
       CV_Assert("Cam open failed");
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    namedWindow("Face Verificaiton Demo", WINDOW_AUTOSIZE);

    FaceVerification *fv = new FaceVerification();
    start = clock();
    while (1){
        cap >> frame;
        flip(frame,frame,1);
        frame.copyTo(output_frame);
        std::vector<Rect> faces;
        int ret = fv->detect(output_frame, faces);
        total_detect_sec = (clock()-start)/(double) CLOCKS_PER_SEC;
        if (total_detect_sec > 2) {
            start = clock();
            fv->reset();
        }
        imshow("Face Verificaiton Demo", output_frame);
        waitKey(33);
   }
}
