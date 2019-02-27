

#include "Optimizer.h"
#include <convert.h>
#include <mappoint.h>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"


int Optimizer::PoseOptimization(frame *pFrame)

{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver= new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(convert::toSE3Quat(pFrame->Tcw));//// 为什么设置两次呢
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    const float deltaMono = sqrt(5.991);
    //const float deltaStereo = sqrt(7.815);


    for(int i=0; i<N; i++)
    {
        mappoint* pMP = pFrame->MapPoints[i];
        if(pMP)
        {
            nInitialCorrespondences++;
            Eigen::Matrix<double,2,1> obs;
            const cv::KeyPoint &kpUn = pFrame->keypoints_l[i];
            obs << kpUn.pt.x, kpUn.pt.y;

            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e->setMeasurement(obs);
            e->setInformation(Eigen::Matrix2d::Identity()) ;///*pFrame->match_score[i]/float(10)

            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
            rk->setDelta(deltaMono);

            e->fx = pFrame->fx;
            e->fy = pFrame->fy;
            e->cx = pFrame->cx;
            e->cy = pFrame->cy;
            cv::Mat Xw = pMP->GetWorldPos();
            e->Xw[0] = Xw.at<float>(0);
            e->Xw[1] = Xw.at<float>(1);
            e->Xw[2] = Xw.at<float>(2);

            optimizer.addEdge(e);
        }

    }
    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );


    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = convert::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);

    return nInitialCorrespondences;
}

