#include <iostream>
#include <opencv2/opencv.hpp>
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

#ifndef _BLUR_DETECTION_H_
#define _BLUR_DETECTION_H_

#include <opencv2/opencv.hpp>

class BlurDetection
{
  public :
    bool checkBlurryImage(std::string img_path, float threshold);

  private :
    void getHaarWavelet(const cv::Mat &src, cv::Mat &dst);
    void getEmax(const cv::Mat &src, cv::Mat &dst, int scale);
};

#endif /* _BLUR_DETECTION_H_ */
