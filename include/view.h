//
// Created by zss on 18-12-20.
//

#ifndef STEREO_VO_VIEW_H
#define STEREO_VO_VIEW_H
#include <opencv2/core/mat.hpp>
#include <pangolin/pangolin.h>
#include <frame.h>


class View
{
public:
    static void DrawGraph(frame &lastframe,frame *currentframe);
    static void DrawMappoints(set<mappoint*> &spRefMPs,int id);

};
#endif //STEREO_VO_VIEW_H
