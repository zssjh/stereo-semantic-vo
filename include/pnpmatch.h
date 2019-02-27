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
    static int poseEstimationPnP(frame *cframe,frame &frame,set<mappoint*> &localmappoints,cv::Mat &mVelocity,cv::Mat &K);
    static void find_feature_matches (
            const cv::Mat& img_1, const cv::Mat& img_2,
            std::vector<cv::KeyPoint>& keypoints_1,
            std::vector<cv::KeyPoint>& keypoints_2,
            std::vector<cv::DMatch >& matches );
    static int poseEstimation2D_2D(frame *CurrentFrame,frame &LastFrame,cv::Mat &K,cv::Mat &fundamental_matrix);

};

#endif
