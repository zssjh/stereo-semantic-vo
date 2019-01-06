//
// Created by zss on 18-12-17.
//

#ifndef STEREO_VO_MAPPOINT_H
#define STEREO_VO_MAPPOINT_H
#include <opencv2/core/core.hpp>
#include <map>
#include <frame.h>
#include <vector>

class frame;
class mappoint
{
public:
    mappoint(cv::Mat &pos,frame *pFrame, int id);//// 构造函数，什么时候需要在cpp中定义{}
    //void computedescriptor();
    cv::Point3f getPosition();
    cv::Mat GetWorldPos();
    void AddObservation( frame *fm, size_t idx);

    cv::Mat ComputeDistinctiveDescriptors();
    std::vector<cv::Mat> GetDescriptors();

public:
    cv::Mat worldpos;///// 利用构造函数pos初始化worldpos
    cv::Mat m_descriptor;
    bool bad;

    //cv::Mat descriptor;
    int observation_num;
    int create_id;
    std::map<frame*,int> observations;

};

#endif
