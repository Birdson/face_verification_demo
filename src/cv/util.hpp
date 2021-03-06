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

#ifndef _UTIL_H_
#define _UTIL_H_

#include <opencv2/opencv.hpp>
#include <dlib/image_io.h>

using namespace cv;

static const std::string KEYWORD_BLURRY = "Too Blurry!";

static const Scalar box_colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)};

static const Scalar landmark_color = CV_RGB(0,255,255);

inline void drawLabel(cv::Mat& img, cv::Rect box, std::string label, Scalar color)
{
  double fontScale = (1.7 * box.width) / 400;
  int id_y = box.y - (fontScale * 30);
  if (id_y < 0) id_y = 0;
  int rect_width = box.width * 0.5;
  if (label.find(KEYWORD_BLURRY) != std::string::npos) {
      rect_width = box.width * 0.8;
      color = Scalar(0, 0, 255);
  }
  cv::rectangle(img, Point(round(box.x), round(id_y)),
        Point(round(box.x + rect_width - 1), round(id_y + (fontScale * 30))),
        color,  CV_FILLED , 8, 0);
  cv::putText(img, label, Point(round(box.x + (fontScale * 5)), round(id_y + (fontScale * 25))),
        cv::FONT_HERSHEY_DUPLEX, fontScale, Scalar(255, 255, 255),
        2);
}

inline void drawFaceBoxes(cv::Mat& img, std::vector<FaceData>& face_datas)
{
  for(unsigned int i = 0; i < face_datas.size(); i++)
  {
    Scalar color = box_colors[i%8];
    cv::rectangle(img, face_datas[i].face_box.tl(), face_datas[i].face_box.br(), color, 6, 8, 0);
    if (face_datas[i].face_id != ""
        && face_datas[i].face_id.find("User#") == std::string::npos
        && face_datas[i].face_id.find(KEYWORD_BLURRY) == std::string::npos) {
      drawLabel(img, face_datas[i].face_box, face_datas[i].face_id, color);
    }
  }
}

inline void drawFaceLandmarks(cv::Mat& img, std::vector<FaceData>& face_datas)
{
  for(unsigned int i = 0; i < face_datas.size(); i++)
  {
    for(unsigned int j = 0; j < face_datas[i].face_landmark.size(); j++) {
      cv::circle(img, face_datas[i].face_landmark[j], 1.3,  landmark_color, 3, 8, 0);
    }
  }
}

inline void drawFPS(cv::Mat& img,  double fps)
{
    char fpsMsg[100];
    Scalar color =  CV_RGB(0,0,255);
    sprintf(fpsMsg, "%.1fFPS", fps);
    cv::putText(img, fpsMsg, cv::Point(5, 30), cv::FONT_HERSHEY_DUPLEX, 1.1, color, 2);
}

inline float getIoU(cv::Rect a, cv::Rect b)
{
  cv::Rect unionRect = a | b;
  cv::Rect intersectionRect = a & b;
  float iou = (float)intersectionRect.area()/(float)unionRect.area();
  return iou;
}

/*inline void CVRect_to_DlibRect(std::vector<dlib::rectangle>& d_rect, std::vector<cv::Rect>& cv_rect)
{
  dlib::rectangle rect;

  d_rect.clear();

  for(unsigned int i=0; i < cv_rect.size(); i++)
  {
    rect.set_left(cv_rect[i].x);
    rect.set_top(cv_rect[i].y);
    rect.set_right(cv_rect[i].x + cv_rect[i].height); //x + height
    rect.set_bottom(cv_rect[i].y + cv_rect[i].width); // y + width
    d_rect.push_back(rect);
  }
}*/

inline void CVRect_to_DlibRect(dlib::rectangle& d_rect, cv::Rect& cv_rect)
{
  d_rect.set_left(cv_rect.x);
  d_rect.set_top(cv_rect.y);
  d_rect.set_right(cv_rect.x + cv_rect.height); //x + height
  d_rect.set_bottom(cv_rect.y + cv_rect.width); // y + width
}

inline void removeDuplicateFaces(std::vector<cv::Rect>& faces)
{
  for(unsigned int i = 0; i < faces.size(); i++) {
    Rect current = faces[i];
    for(unsigned int j = i; j < faces.size(); j++) {
      if(j == i) {
        continue;
      } else {
        Rect temp = faces[j];
        if(getIoU(current, temp) > 0.1) {
          vector<Rect>::iterator iter = faces.begin() + j;
          faces.erase(iter);
          j--;
        }
      }
    }
  }
}

inline void sortFaces(std::vector<cv::Rect>& faces, int max_num)
{
  cv::Rect temp;

  if (faces.size() > max_num) {
    for (unsigned int i = 0; i < faces.size() - 1; i++) {
      for (unsigned int j = 1; j < faces.size() - i; j++) {
        if (faces[j - 1].area() < faces[j].area()) {
          temp = faces[j - 1];
          faces[j - 1] = faces[j];
          faces[j] = temp;
        }
      }
    }

    std::vector<cv::Rect> temp_faces;
    for(unsigned int i = 0; i < max_num; i++) {
      temp_faces.push_back(faces[i]);
    }
    faces = temp_faces;
  }

  for (unsigned int i = 0; i < faces.size() - 1; i++) {
    for (unsigned int j = 1; j < faces.size() - i; j++) {
      if (faces[j - 1].x > faces[j].x) {
        temp = faces[j - 1];
        faces[j - 1] = faces[j];
        faces[j] = temp;
      }
    }
  }
}

#endif /* _UTIL_H_ */
