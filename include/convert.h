//
// Created by zss on 18-12-17.
//

#ifndef STEREO_VO_CONVERT_H
#define STEREO_VO_CONVERT_H
#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

class convert
{
public:
    static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
    static cv::Mat R_t_to_Twc(const cv::Mat &R,const cv::Mat &t);
    static cv::Mat R_t_to_Tcw(const cv::Mat &R,const cv::Mat &t);
    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
    static std::vector<float> toQuaternion(const cv::Mat &M);

};

#endif

