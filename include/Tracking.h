#ifndef STEREO_VO_TRACK_H
#define STEREO_VO_TRACK_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <pangolin/pangolin.h>

#include <mutex>
#include <string.h>
#include <frame.h>
#include <fstream>
#include <set>
#include "semantic.h"
using namespace std;
//class keyframe;

class Semantic;
class Tracking
{  

public:
    Tracking(const string &strSettingPath);
    void init();
    void Track( cv::Mat &imLeft,  cv::Mat &imRight,cv::Mat &imdepth,double &timestamp,cv::Mat &K, float &bf,ofstream &f,ofstream &f2,pangolin::OpenGlMatrix &Twc_M);
    void Tracklastframe();
    void Tracklocalmap();
    void SaveTrajectoryAndDraw(ofstream &f,ofstream &f2);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    frame* Getcurrentframe();
    void Setsemanticer(Semantic* ser);
    //bool needkeyframe();
    //void createkeyframe();

public:
    Semantic* Semanticer;
    frame lastframe;
    frame *currentframe;
    static int frame_num;
    //keyframe referenceKF;
    //keyframe lastKF;
    cv::Mat K;
    float bf;
    bool local;
    static std::set<mappoint*> LocalMapPoints;
};

#endif

