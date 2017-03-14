#include "config_reader.h"
#include "face_verification_demo.hpp"
#include "cv/util.hpp"
#include "V4L2Device.h"

static const int INTERPOLATION = cv::INTER_NEAREST;

WebCamCap * WebCamCap::s_instance = NULL;
WebCamCap* WebCamCap::instance() {
    if (s_instance == NULL)
        s_instance = new WebCamCap();
    return s_instance;
}

void WebCamCap::ProcessThread(WebCamCap * cap) {
    cap->processFrame();
}

void *detectInThread(void *ptr) {
    WebCamCap::instance()->ProcessThread(WebCamCap::instance());
    return 0;
}

void WebCamCap::startVideoCapture() {
    if (fv == NULL)
        fv = new FaceVerification();
    namedWindow("Face Verificaiton Demo", 1);
    const int skip_frames = ConfigReader::getInstance()->cv_config.skip_frames;
    const Size webcamSize(ConfigReader::getInstance()->webcam_config.width, ConfigReader::getInstance()->webcam_config.height);

    if (!ConfigReader::getInstance()->test_config.enable_image_test) {
        if (ConfigReader::getInstance()->test_config.enable_video_test) {
            cap.open(ConfigReader::getInstance()->test_config.video_test_path);
            if (!cap.isOpened()) {
                keepRunning = false;
                // check if succeeded to openc the video file
                CV_Assert("VideoFile open failed");
            }
        } else {
            cap.open(ConfigReader::getInstance()->webcam_config.device);
            if (!cap.isOpened()) {
                keepRunning = false;
                // check if succeeded to connect to the camera
                CV_Assert("WebCam open failed");
            }
            cap.set(CV_CAP_PROP_FRAME_WIDTH, webcamSize.width);
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, webcamSize.height);
            cap.set(CV_CAP_PROP_AUTOFOCUS, 0);
        }
    }

    V4L2Device v4l2device;
    if (ConfigReader::getInstance()->webcam_config.enable_virtual_device) {
        v4l2device.openV4L2Device(ConfigReader::getInstance()->webcam_config.width,
                            ConfigReader::getInstance()->webcam_config.height + 320,
                            ConfigReader::getInstance()->webcam_config.virtual_device_path);
    }

    // Inside loop.
    cv::Mat frame;
    cv::Mat image_frame;
    cv::Mat out_frame;
    cv::Mat virtual_out_frame;
    clock_t start = clock();
    clock_t time = clock();
    faces.clear();
    face_ids.clear();
    landmarks.clear();
    int move_point = 0;
    while (keepRunning) {
        frame_count++;
        if (ConfigReader::getInstance()->test_config.enable_image_test) {
            image_frame = imread(ConfigReader::getInstance()->test_config.image_test_path);
            cv::Rect area(move_point, move_point, image_frame.cols - move_point, image_frame.rows - move_point);
            image_frame = image_frame(area);

            if (image_frame.cols != webcamSize.width || image_frame.rows != webcamSize.height)
                resize(image_frame, image_frame, webcamSize, 0, 0, INTERPOLATION);

            image_frame.copyTo(frame);

            move_point++;
            move_point = move_point % 250;
        } else {
            cap >> frame;
        }

        if(frame.empty()){
            if (ConfigReader::getInstance()->test_config.enable_video_test) {
                cap.release();
                cap.open(ConfigReader::getInstance()->test_config.video_test_path);
                if (!cap.isOpened()) {
                    // check if succeeded to openc the video file
                    CV_Assert("VideoFile open failed");
                    keepRunning = false;
                    break;
                }
                cap >> frame;
            } else {
                keepRunning = false;
                break;
            }
        }

        if (!ConfigReader::getInstance()->test_config.enable_image_test
            && !ConfigReader::getInstance()->test_config.enable_video_test) {
            if (frame.cols != webcamSize.width || frame.rows != webcamSize.height)
                resize(frame, frame, webcamSize, 0, 0, INTERPOLATION);
            flip(frame,frame,1);
        }

#ifdef ENABLE_MULTI_THREAD
        if (frame_buffer.size() <= 1) {
            frame_buffer.push_back(frame);
        } else {
            frame_buffer.erase(frame_buffer.begin());// this part deletes the first element
            frame_buffer.push_back(frame);
        }
        if (!is_start_processing_frame) {
            is_start_processing_frame = true;
            if(pthread_create(&detectThread, 0, detectInThread, 0))
                CV_Error(Error::Code::StsError, "Thread(detect) creation failed");
            pthread_detach(detectThread);
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
            total_time = clock()-start;
            fps = (double) CLOCKS_PER_SEC / (total_time / frame_count);
            frame_count = 0;
            start = clock();
        }

        cv::Mat combine;
        fv->createFaceWindow(frame, combine, faces);

        if (faces.size() > 0 && ConfigReader::getInstance()->cv_config.enable_draw_face_boxs) {
            drawFaceBoxes(frame, faces, face_ids);
        }

        if (landmarks.size() > 0 && ConfigReader::getInstance()->cv_config.enable_draw_face_landmarks) {
            drawFaceLandmarks(frame, landmarks);
        }

        if (ConfigReader::getInstance()->cv_config.enable_draw_face_landmarks) {
            drawFPS(frame, fps);
        }

        const int out_frame_hight = frame.rows + max(combine.rows, 320);
        const int out_frame_width = max(frame.cols, combine.cols);
        out_frame = Mat::zeros(out_frame_hight, out_frame_width, frame.type());
        combine.copyTo(out_frame(Rect(0, max(320 - combine.rows, 0), combine.cols, combine.rows)));
        frame.copyTo(out_frame(Rect(0, max(combine.rows, 320), frame.cols, frame.rows)));
        combine.release();

        imshow("Face Verificaiton Demo", out_frame);

        //imshow("Face Verificaiton Demo", frame);

        if (ConfigReader::getInstance()->webcam_config.enable_virtual_device) {
            cv::cvtColor(out_frame, virtual_out_frame, CV_BGR2YUV_I420);
            flip(virtual_out_frame,virtual_out_frame,1);
            v4l2device.writeFrame(virtual_out_frame.data);
        }

        if (waitKey(30) >= 0) {
            break;
        }
    }
    frame.release();
    out_frame.release();
    virtual_out_frame.release();
    cap.release();
    delete fv;
}

#ifdef ENABLE_MULTI_THREAD
void WebCamCap::processFrame() {
  while (keepRunning) {
    if (fv != NULL && !frame_buffer.empty()) {
        int ret = fv->detect(frame_buffer.back(), faces, face_ids, landmarks);
        if (ret == -1) {
          keepRunning = false;
          break;
        }
    }
  }
}
#else
void WebCamCap::processFrame() {
    if (fv != NULL && !frame_buffer.empty()) {
        int ret = fv->detect(frame_buffer.back(), faces, face_ids, landmarks);
        if (ret == -1) {
            keepRunning = false;
        }
    }
}
#endif

int main(int argc, char** argv) {
    if(ConfigReader::getInstance()->initConfig()) {
        WebCamCap::instance()->startVideoCapture();
    }
    return 0;
}
