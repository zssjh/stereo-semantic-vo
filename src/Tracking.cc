
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
    Velocity=cv::Mat::eye(4,4,CV_32F);

    for (int i = 0; i <currentframe->N ; ++i) {
        const float u = currentframe->keypoints_l[i].pt.x;
        const float v = currentframe->keypoints_l[i].pt.y;
        const float z = currentframe->depthimg.at<float>(v,u);

        if(currentframe->offline_box.size()!=0)
        {
            for (int j = 0; j < currentframe->offline_box.size(); ++j) {

                int left=currentframe->offline_box[j][0];
                int right=currentframe->offline_box[j][1];
                int top=currentframe->offline_box[j][2];
                int bottom=currentframe->offline_box[j][3];
                if(u>left-5&&u<right+5&&v>top-5&&v<bottom+5)
                {
                    dynamic=true;
                    break;
                }
            }
         }

        //// disable following lines if using offline semantic ////
//        if(currentframe->boxes.size()!=0)
//        {
//            for (int j = 0; j < currentframe->boxes.size(); ++j) {
//
//                int x0=currentframe->boxes[j].tl().x;
//                int y0=currentframe->boxes[j].tl().y;
//                int x1=currentframe->boxes[j].br().x;
//                int y1=currentframe->boxes[j].br().y;
//                if(u>x0&&u<x1&&v>y0&&v<y1)
//                {
//                    currentframe->DY_keypoints.push_back(cv::Point2f(u,v));
//                    dynamic=true;
//                    break;
//                }
//            }
//         }
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
}

void::Tracking::GetVelocity()
{
    cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
    lastframe.Rwc.copyTo(LastTwc.rowRange(0,3).colRange(0,3));
    lastframe.twc.copyTo(LastTwc.rowRange(0,3).col(3));
    Velocity = currentframe->Tcw*LastTwc;

}
void Tracking::Tracklastframe()
{
    if(frame_num==0)
        init();

    else
        pnpmatch::poseEstimationPnP(currentframe,lastframe,LocalMapPoints,Velocity,K);


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
//        View::DrawMappoints(LocalMapPoints,frame_num);
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


 void Tracking::Track( cv::Mat &imLeft, cv::Mat &imRight,cv::Mat &imdepth, cv::Mat &img_detect,double &timestamp,cv::Mat &K,
                       float &bf,ofstream &f,ofstream &f2,pangolin::OpenGlMatrix &Twc_M,vector<vector<int>> &detection_box)
{

    currentframe=new frame(imLeft,imRight,imdepth,img_detect,timestamp,K,bf,detection_box);

    //// disable this line if using offline semantic ////
//    Semanticer->Insertframe(currentframe);

    //// LK tracking ////
//
//    cout<<"上一帧："<<lastframe.DY_keypoints.size()<<endl;
//    if(!lastframe.DY_keypoints.empty())
//        cv::calcOpticalFlowPyrLK(lastframe.leftimg,currentframe->leftimg,lastframe.DY_keypoints,currentframe->LK_keypoints,lastframe.status,lastframe.error);
//
//
//    for ( auto kp:currentframe->LK_keypoints )
//        cv::circle(currentframe->leftimg, kp, 10, cv::Scalar(0, 240, 0), 1);/// 不直接在原图上画
//
//    for ( auto kp2:lastframe.DY_keypoints )
//        cv::circle(lastframe.leftimg, kp2, 10, cv::Scalar(0, 240, 0), 1);/// 不直接在原图上画
//
//
//    int ii=0;
//    int count=0;
//    for ( auto iter=lastframe.DY_keypoints.begin(),iter2=currentframe->LK_keypoints.begin(); iter!=lastframe.DY_keypoints.end(),iter2!=currentframe->LK_keypoints.end(); ii++)
//    {
//        if ( lastframe.status[ii] == 0 )
//        {
//            iter = lastframe.DY_keypoints.erase(iter);
//            iter2 = currentframe->LK_keypoints.erase(iter2);
//            count++;
//            continue;
//        }
//        iter++;
//        iter2++;
//    }
//
//    cout<<"删除："<<count<<endl;
//
//
//    for (auto kp:currentframe->LK_keypoints) {
//        currentframe->DY_keypoints.push_back(kp);
//    }

    currentframe->featuredetect(imLeft);
    currentframe->dispimg=currentframe->ElasMatch(imLeft,imRight);//// SGBM计算的CV_16s，转化为CV_8U,深度滤波???
    currentframe->computekeypoint_r();
    currentframe->disp2Depth(bf);
    currentframe->id=frame_num;

    Tracklastframe();

    SaveTrajectoryAndDraw(f,f2);
    GetCurrentOpenGLCameraMatrix(Twc_M);
     if(frame_num>0)
         GetVelocity();
    lastframe = frame(currentframe);
    lastframe.createmappoint(LocalMapPoints);
     if(frame_num>=4)
     {
         for (set<mappoint*>::iterator it = LocalMapPoints.begin(); it != LocalMapPoints.end(); )
         {
             mappoint *mp_2=*it;
             if(mp_2->create_id<=frame_num-4)
                 LocalMapPoints.erase(it++);//// 不能使用删除*it,也不能在for加it++,必须分着写it++
                 ////  *it = static_cast<mappoint*>(NULL);好像更节约时间
             else
                 it++;
         }
     }
    frame_num++;
}




