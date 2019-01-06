//
// Created by zss on 18-12-17.
//

#ifndef STEREO_VO_PNP_H
#define STEREO_VO_PNP_H

#include <opencv2/core/mat.hpp>
#include <sophus/se3.h>
#include <frame.h>
#include <set>

class pnpmatch
{
public:
    static cv::Mat Cur_Tcw;
public:
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    static int poseEstimationPnP(frame *cframe,frame &frame,cv::Mat &K);
    static int poseEstimationlocalmap(frame *CurrentFrame,set<mappoint*> &localmappoints,cv::Mat &K);

};

#endif
