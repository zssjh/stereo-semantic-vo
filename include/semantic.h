//
// Created by zss on 19-1-3.
//

#ifndef STEREO_VO_SEMANTIC_H
#define STEREO_VO_SEMANTIC_H

#include <opencv2/core/mat.hpp>
#include "YOLOv3SE.h"
#include "string.h"
#include "Tracking.h"

using namespace std;
class Tracking;
class Semantic
{
public:
    Semantic(vector<string> &imleft,vector<string> &imright);
    void Run();
    void SetTracker(Tracking *pTracker);
    void Insertframe(frame *cur_frame);
    bool Checknewframe();
    YOLOv3 detector;

public:
    vector<string> left;
    vector<string> right;
    Tracking* mpTracker;
    std::list<frame*> NewFrames;
};

#endif //STEREO_VO_SEMANTIC_H
