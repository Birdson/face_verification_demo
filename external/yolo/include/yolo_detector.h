#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include "box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
    int xmin, ymin, width, height;
} face;

bool yolo_init(char *cfgfile, char *weightfile);

face *yolo_detect(IplImage *frame, float thresh, float hier_thresh, float scale);

int yolo_get_detection_num();

bool sub_yolo_init(char *cfgfile, char *weightfile);

face *sub_yolo_detect(IplImage *frame, float thresh, float hier_thresh, float scale);

int sub_yolo_get_detection_num();

#ifdef __cplusplus
}
#endif

#endif
