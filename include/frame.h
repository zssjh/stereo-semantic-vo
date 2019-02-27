//
// Created by zss on 18-12-15.
//
#ifndef STEREO_VO_FRAME_H
#define STEREO_VO_FRAME_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>  //cvtcolor
#include<opencv2/features2d/features2d.hpp>
#include <vector>
#include <mappoint.h>
#include <set>
#include <list>
#include <Thirdparty/libelas/src/elas.h>
#include "YOLOv3SE.h"


using namespace std;

//class keyframe;
class mappoint;
class frame
{
public: ////不声明默认是private
     frame();
     frame(frame *frame);//// 这个参数必须是const类型，否则会找不到正确的构造函数定义
     frame( cv::Mat &imLeft,  cv::Mat &imRight, cv::Mat &imdepth,cv::Mat &img_detect,double &timestamp,cv::Mat &K,float &bf,vector<vector<int>> &detection_box);

     void SetPose(cv::Mat mTcw);
     void featuredetect( cv::Mat &img);
     cv::Mat ElasMatch( cv::Mat &leftImage,cv::Mat &rightImage);
    cv::Mat ElasMatch2( cv::Mat &leftImage,cv::Mat &rightImage);
     void disp2Depth( float bf);
     cv::Mat UnprojectStereo(const float &u,const float &v,const float &z);
     void createmappoint(set<mappoint*> &localmap);
     void computekeypoint_r();

public:
    int N;
    double timestamp;
    long int id;
    cv::Mat leftimg,rightimg;
    cv::Mat dispimg,depthimg;
    cv::Mat detectimg;
    vector<cv::KeyPoint> keypoints_l;
    vector<cv::Point2f> keypoints_r;
    cv::Mat f_descriptor;
    vector<mappoint*> MapPoints;
    vector<float> match_score;
    vector<bool> inlier;
    std::vector<BoxSE> boxes;
    bool have_detected;
    vector<unsigned char> status;
    vector<float> error;
    //keyframe referenceKF;
    vector<vector<int>> offline_box;
    float width;
    float height;

    //// 相机内参
    cv::Mat K;
    float fx;
    float fy;
    float cx;
    float cy;
    float bf;

    cv::Mat Tcw;
    cv::Mat Twc;
    cv::Mat tcw;
    cv::Mat twc;
    cv::Mat Rcw;
    cv::Mat Rwc;

};

#endif