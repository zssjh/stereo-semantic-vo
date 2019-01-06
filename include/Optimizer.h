#ifndef STEREO_VO_OP_H
#define STEREO_VO_OP_H

#include <frame.h>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

class Optimizer
{
public:
    int static PoseOptimization(frame *pFrame);

};

#endif

