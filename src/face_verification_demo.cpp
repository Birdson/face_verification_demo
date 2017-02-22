#include "config_reader.h"
#include "face_verification_demo.hpp"
#include "cv/util.hpp"

WebCamCap * WebCamCap::s_instance = NULL;
WebCamCap* WebCamCap::instance() {
    if (s_instance == NULL)
        s_instance = new WebCamCap();
    return s_instance;
}

void WebCamCap::CaptureThread(WebCamCap * cap) {
    cap->captureFrame();
}

void WebCamCap::ProcessThread(WebCamCap * cap) {
    cap->processFrame();
}

void WebCamCap::captureFrame() {
    if (fv == NULL)
        fv = new FaceVerification();
    namedWindow("Face Verificaiton Demo", 1);
    const int skip_frames = ConfigReader::getInstance()->cv_config.skip_frames;

    cap.open(ConfigReader::getInstance()->webcam_config.device);
    if (!cap.isOpened())  // check if succeeded to connect to the camera
        CV_Assert("WebCam open failed");

    cap.set(CV_CAP_PROP_FRAME_WIDTH, ConfigReader::getInstance()->webcam_config.width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, ConfigReader::getInstance()->webcam_config.height);

    // Inside loop.
    cv::Mat frame;
    clock_t start = clock();
    clock_t time = clock();
    faces.clear();
    face_ids.clear();
    landmarks.clear();
    while (keepRunning) {
        frame_count++;
        cap >> frame;
        flip(frame,frame,1);
#ifdef ENABLE_MULTI_THREAD
        if (frame_buffer.size() <= 1) {
            frame_buffer.push_back(frame);
        } else {
            frame_buffer.erase(frame_buffer.begin());// this part deletes the first element
            frame_buffer.push_back(frame);
        }
        if (!is_processing_frame) {
            is_processing_frame = true;
            instance()->processThread = thread(ProcessThread, instance());
            instance()->processThread.detach();
        }
#else
        if (frame_buffer.size() <= 1) {
            frame_buffer.push_back(frame);
        } else {
            if ((int)frame_count%skip_frames == 0) {
                processFrame();
                frame_buffer.clear();
            }
        }
#endif

        total_time = (clock()-time)/(double) CLOCKS_PER_SEC;
        if (total_time > 2.0) {
            time = clock();
            fv->reset();
        }

        if (frame_count == 30) {
            total_time = 1000.0 * (clock()-start)/(double) CLOCKS_PER_SEC;
            fps = 1000 / (total_time / frame_count);
            frame_count = 0;
            start = clock();
        }

        cv::Mat combine;
        fv->showFaceWindow(frame, combine, faces);

        if (faces.size() > 0 && ConfigReader::getInstance()->cv_config.enable_draw_face_boxs) {
            drawFaceBoxes(frame, faces, face_ids);
        }

        if (landmarks.size() > 0 && ConfigReader::getInstance()->cv_config.enable_draw_face_landmarks) {
            drawFaceLandmarks(frame, landmarks);
        }

        if (ConfigReader::getInstance()->cv_config.enable_draw_face_landmarks) {
            drawFPS(frame, fps);
        }

        Mat out_frame = Mat::zeros(frame.rows+combine.rows, frame.cols, frame.type());
        combine.copyTo(out_frame(Rect(0, 0, combine.cols, combine.rows)));
        frame.copyTo(out_frame(Rect(0, combine.rows, frame.cols, frame.rows)));

        imshow("Face Verificaiton Demo", out_frame);

        //imshow("Face Verificaiton Demo", frame);

        if (waitKey(30) >= 0) {
            break;
        }
    }
    frame.release();
    cap.release();
    delete fv;
}

void WebCamCap::processFrame() {
    if (fv != NULL && !frame_buffer.empty()) {
        int ret = fv->detect(frame_buffer.back(), faces, face_ids, landmarks);
        if (ret == -1) {
          keepRunning = false;
        }
    }
    is_processing_frame = false;
}

int main(int argc, char** argv) {
    if(ConfigReader::getInstance()->initConfig()) {
        //WebCamCap::instance()->processThread = thread(&WebCamCap::ProcessThread, WebCamCap::instance());
        //WebCamCap::instance()->processThread.detach();
        WebCamCap::instance()->captureThread = thread(&WebCamCap::CaptureThread, WebCamCap::instance());
        WebCamCap::instance()->captureThread.join();
        //const float threshold = ConfigReader::getInstance()->yolo_config.confidence_threshold;
        //const float sub_threshold = ConfigReader::getInstance()->yolo_config.sub_confidence_threshold;
        //printf("threshold is %f, sub threshold is %f", threshold, sub_threshold);
    }
    return 0;
}
