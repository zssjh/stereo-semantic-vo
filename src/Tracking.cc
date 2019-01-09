
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Tracking.h>
#include <Optimizer.h>
#include <pnpmatch.h>
#include <opencv2/core/affine.hpp>
#include <pangolin/pangolin.h>
#include <view.h>
#include <iomanip>
#include <convert.h>
#include <opencv/cv.hpp>
#include <include/semantic.h>

using namespace std;

////static 变量初始化
int Tracking::frame_num=0;
std::set<mappoint*> Tracking::LocalMapPoints;

Tracking::Tracking(const string &strSettingPath):local(false)//// 成员变量必须初始化
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat bK = cv::Mat::eye(3,3,CV_32F);
    bK.at<float>(0,0) = fx;
    bK.at<float>(1,1) = fy;
    bK.at<float>(0,2) = cx;
    bK.at<float>(1,2) = cy;
    bK.copyTo(K);

    bf = fSettings["Camera.bf"];

}

void Tracking::init()
{
    bool dynamic=false;
    currentframe->SetPose(cv::Mat::eye(4,4,CV_32F));

    for (int i = 0; i <currentframe->N ; ++i) {
        const float u = currentframe->keypoints_l[i].pt.x;
        const float v = currentframe->keypoints_l[i].pt.y;
        const float z = currentframe->depthimg.at<float>(v,u);

        if(currentframe->boxes.size()!=0)
        {
            for (int j = 0; j < currentframe->boxes.size(); ++j) {

                int x0=currentframe->boxes[j].tl().x;
                int y0=currentframe->boxes[j].tl().y;
                int x1=currentframe->boxes[j].br().x;
                int y1=currentframe->boxes[j].br().y;
                if(u>x0&&u<x1&&v>y0&&v<y1)
                {
                    dynamic=true;
                    break;
                }
            }
         }
        if(z>0&&!dynamic)
        {

            cv::Mat x3D=currentframe->UnprojectStereo(u,v,z);
            mappoint* newmp=new mappoint(x3D,currentframe,i);////什么时候定义*,*的时候是->
            newmp->AddObservation(currentframe,i);
            newmp->create_id=currentframe->id;
            LocalMapPoints.insert(newmp);
            //newmp->m_descriptor=currentframe.f_descriptor.row(i);
            currentframe->MapPoints[i]=newmp;
        }
    }

    lastframe=frame(currentframe);
}

//bool Tracking::needkeyframe()
//{
//    if(frame_num%10==0)
//        return true;
//}

//void Tracking::createkeyframe()//// 创建关键帧，为关键帧初始化新的地图点，之前有匹配上的不用，保证每一个特征点对应一个地图点
//{
//
//    keyframe pKF = keyframe(currentframe,kf_num);
//    kf_num++;
//    referenceKF = pKF;
//    currentframe.referenceKF=pKF;
//
//    //// mCurrentFrame.UpdatePoseMatrices();
//    for (int i = 0; i <currentframe.N ; ++i) {
//        if(!currentframe.MapPoints[i]&&currentframe.MapPoints[i]->observation_num>2)
//            continue;
//
//        currentframe.MapPoints[i]=NULL;//// 清除有地图点但是观测次数少的
//        const float u = currentframe.keypoints_l[i].pt.x;
//        const float v = currentframe.keypoints_l[i].pt.y;
//        const float z = currentframe.depthimg.at<float>(u,v);
//
//        if(z>0)
//        {
//            cv::Mat x3D=currentframe.UnprojectStereo();
//            mappoint *newmp=new mappoint(x3D,i);
//            newmp->computedescriptor();
//            Map_.insertmappoint(newmp);
//            currentframe.MapPoints[i]=newmp;
//        }
//
//    }
//    lastKF = pKF;
//}

void Tracking::Tracklocalmap()
{
    pnpmatch::poseEstimationlocalmap(currentframe,LocalMapPoints,K);

}

void Tracking::Tracklastframe()
{
    float inlier_ratio=0;
    if(frame_num==0)
        init();

    else
//        Tracklocalmap();
        pnpmatch::poseEstimationPnP(currentframe,lastframe,K);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    vector<int> pts2d_index;

    for (int j = 0; j <currentframe->N ; ++j) {
        mappoint *mapp=currentframe->MapPoints[j];
        if(mapp)
        {
            pts2d_index.push_back(j);
            pts2d.push_back ( currentframe->keypoints_l[j].pt );
            pts3d.push_back( cv::Point3f( mapp->worldpos.at<float>(0,0), mapp->worldpos.at<float>(1,0), mapp->worldpos.at<float>(2,0)));
        }
    }

    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );////double
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    int num_inliers_ = inliers.rows;
    inlier_ratio=(float)num_inliers_/(float)pts2d.size();

//    if(inlier_ratio<0.2)
//        cout<<"内点比例："<<inlier_ratio<<"  "<<num_inliers_<<"  "<<pts2d.size()<<endl;
    cv::Mat Cur_Rcw(3,3,CV_32F);
    cv::Rodrigues(rvec, Cur_Rcw);

    cv::Mat Tcl;
    Tcl=(cv::Mat_<float>(4,4)<<
                             Cur_Rcw.at<double>(0,0),Cur_Rcw.at<double>(0,1),Cur_Rcw.at<double>(0,2),tvec.at<double>(0,0),
            Cur_Rcw.at<double>(1,0),Cur_Rcw.at<double>(1,1),Cur_Rcw.at<double>(1,2),tvec.at<double>(1,0),
            Cur_Rcw.at<double>(2,0),Cur_Rcw.at<double>(2,1),Cur_Rcw.at<double>(2,2),tvec.at<double>(2,0),
            0,0,0,1);

    currentframe->SetPose(Tcl*currentframe->Tcw);

    Optimizer::PoseOptimization(currentframe);
}


void  Tracking::SaveTrajectoryAndDraw(ofstream &f,ofstream &f2)
{
    cv::Mat R = currentframe->Rwc;
    vector<float> q = convert::toQuaternion(R);
    cv::Mat t = currentframe->twc;
    f2 << setprecision(6) << currentframe->timestamp << setprecision(7) << " " << t.at<float>(0,0) << " " << t.at<float>(1,0) << " " << t.at<float>(2,0)
      << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    f << setprecision(9) << currentframe->Rwc.at<float>(0,0) << " " << currentframe->Rwc.at<float>(0,1)  << " "
      << currentframe->Rwc.at<float>(0,2) << " "  << currentframe->twc.at<float>(0) << " " <<
      currentframe->Rwc.at<float>(1,0) << " " << currentframe->Rwc.at<float>(1,1)  << " " <<
      currentframe->Rwc.at<float>(1,2) << " "  << currentframe->twc.at<float>(1) << " " <<
      currentframe->Rwc.at<float>(2,0) << " " << currentframe->Rwc.at<float>(2,1)  << " " <<
      currentframe->Rwc.at<float>(2,2) << " "  << currentframe->twc.at<float>(2) << endl;

    if(frame_num>0)
    {
        View::DrawGraph(lastframe,currentframe);
//        View::DrawMappoints(LocalMapPoints);
        pangolin::FinishFrame();
    }
}

void Tracking::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!currentframe->Twc.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);

        Rwc = currentframe->Rwc;
        twc = currentframe->twc;
        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

frame* Tracking::Getcurrentframe()
{
    return currentframe;
}

void Tracking::Setsemanticer(Semantic* ser)
{
    Semanticer=ser;
}

 void Tracking::Track( cv::Mat &imLeft, cv::Mat &imRight,cv::Mat &imdepth, double &timestamp,cv::Mat &K, float &bf,ofstream &f,ofstream &f2,pangolin::OpenGlMatrix &Twc_M)
{

    currentframe=new frame(imLeft,imRight,imdepth,timestamp,K,bf);

    Semanticer->Insertframe(currentframe);

    currentframe->featuredetect(imLeft);
    currentframe->dispimg=currentframe->ElasMatch(imLeft,imRight);//// SGBM计算的CV_16s，转化为CV_8U
    currentframe->computekeypoint_r();
    currentframe->disp2Depth(bf);
    currentframe->id=frame_num;

    Tracklastframe();

    SaveTrajectoryAndDraw(f,f2);
    GetCurrentOpenGLCameraMatrix(Twc_M);

    lastframe = frame(currentframe);
    lastframe.createmappoint(LocalMapPoints);
    frame_num++;

}




