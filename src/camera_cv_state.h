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

#ifndef _CAMERA_CV_STATE_H_
#define _CAMERA_CV_STATE_H_

enum CameraCvState
{
  CAMERA_CV_NONE    = 0x00,

  CV_INIT,
  CV_FR_INIT,
  CV_OT_INIT,
  CV_GR_INIT,
  CV_RESET,
  CV_READY,

  CAMERA_INIT,
  CAMERA_RESET,
  CAMERA_READY,

  CV_FR_INIT_FAIL,
  CV_FR_REGISTER,
  CV_FR_DETECTING,
  CV_FR_DETECT,
  CV_FR_VERIFY,
  CV_FR_HIT,
  CV_FR_NONE,
  CV_FR_MOVE_TIMEOUT,

  CV_OT_TRACKING,
  CV_OT_DETECT,
  CV_OT_NOT_CENTROL,
  CV_OT_MISS,
  CV_OT_READY_TO_SHOOT,

  CV_GR_DETECTING,
  CV_GR_HIT,

  CAMERA_TAKING_PICTURE,
  CAMERA_PICTURE_READY

};

enum CameraShootMode
{
  TAKE_SINGEL_PICTURE  = 0X00,
  TAKE_5_PICTURE,
  TAKE_360_PICUTE,
  TAKE_CYCLE_PICTUE,
  TAKE_VIDEO
};

static const char* CameraCvStateStr[]
{
  "camera cv none",
  
  "cv initing",
  "cv face recognation initing",
  "cv object track initing",
  "cv gesture recognation initing",
  "cv reset",
  "cv ready",

  "camera initing",
  "camera reseting",
  "camera ready",

  "cv face recognation failed",
  "cv register face target",
  "cv face detecting",
  "cv face detected",
  "cv face verified",
  "cv face target hit",
  "cv target face miss",
  "cv face recognation wait move timeout",

  "cv object tracking",
  "cv object track detecting",
  "cv object not in image centrol",
  "cv object miss",
  "cv object in centrol and ready to shoot",

  "cv gesture detecting",
  "cv gesture hit",

  "camera taking picture",
  "camera taking picture ready"
};

#endif /* _CAMERA_CV_STATE_H_ */
