//
// Created by zss on 18-12-17.
//

#include <map.h>

Map::Map()
{

}

void Map::insertmappoint(mappoint* newmp)
{
    Mp_s.insert(newmp);
}

//void Map::insertkeyframe(keyframe &newKF)
//{
//    Keyframes.insert(newKF);
//}