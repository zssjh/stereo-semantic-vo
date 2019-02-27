//
// Created by zss on 18-12-17.
//

#ifndef STEREO_VO_MAP_H
#define STEREO_VO_MAP_H

#include <set>
#include <mappoint.h>
//class keyframe;
//class mappoint;
class Map
{
public:
    Map();
    void insertmappoint(mappoint* newmp);//// 形参是指针，引用的区别
public:
    std::set<mappoint*> Mp_s;
};

#endif


