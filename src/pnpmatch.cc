//
// Created by zss on 18-12-17.
//

#include <pnpmatch.h>
#include <opencv/cv.hpp>
#include <convert.h>
#include <opencv/cxeigen.hpp>
#include <Tracking.h>

cv::Mat pnpmatch::Cur_Tcw;
int pnpmatch::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)/// h文件中的static，在cpp文件中不用加
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


int pnpmatch::poseEstimationPnP(frame *CurrentFrame,frame &LastFrame,set<mappoint*> &localmappoints,cv::Mat &mVelocity,cv::Mat &K)
{
    float inlier_ratio=0;
    cv::Mat outImg(CurrentFrame->detectimg.rows+LastFrame.detectimg.rows+1,CurrentFrame->detectimg.cols,CurrentFrame->detectimg.type());
    vector<cv::Point2f> cur_po;
    vector<cv::Point2f> last_po;

    vector<cv::Point2f> cur_pro;
    vector<cv::Point2f> last_fea;

    vector<int > cur_score;
    vector<float> cur_gradient;



    /// 在跟踪之前设置当前帧的位姿初始值是上一阵的位姿
//    CurrentFrame->SetPose(mVelocity*LastFrame.Tcw);
//    cout<<mVelocity<<endl;

    cv::Mat Rcw=CurrentFrame->Rcw;
    cv::Mat tcw=CurrentFrame->tcw;

    int matchess=0;
    int good=0;
    int k_bad=0;
    int class_bad=0;
    int zzz=0;
    int uuu=0;
    int vvv=0;

    for (int i = 0; i <LastFrame.N ; ++i) {
        mappoint *mp=LastFrame.MapPoints[i];

        if(mp&&!mp->bad)
        {
            cv::circle(LastFrame.detectimg,LastFrame.keypoints_l[i].pt,5,cv::Scalar(0,0,256),-1);
            cv::Mat Pw=mp->worldpos;
            cv::Mat Pc=Rcw*Pw+tcw;
            float fx=K.at<float>(0,0);
            float fy=K.at<float>(0,2);
            float cx=K.at<float>(1,1);
            float cy=K.at<float>(1,2);

            if(Pc.at<float>(2)<0)
            {
                zzz++;
//                continue;
            }

            float u=fx*Pw.at<float>(0,0)/Pw.at<float>(2,0)+cx;
            float v=fy*Pw.at<float>(1,0)/Pw.at<float>(2,0)+cy;

            if(u<0||u>CurrentFrame->width)
            {
                uuu++;
//                continue;
            }



            if(v<0||v>CurrentFrame->height)
            {
               vvv++;
//                continue;
            }

            last_fea.push_back(LastFrame.keypoints_l[i].pt);
            cur_pro.push_back(cv::Point2f(u,v));
//            cv::circle(CurrentFrame->detectimg,cv::Point2f(u,v),10,cv::Scalar(0,256,0),1);

            bool dynamic=false;
            vector<cv::KeyPoint> KP=CurrentFrame->keypoints_l;
            vector<cv::Point2f> KP_r=CurrentFrame->keypoints_r;

            cv::Mat dMP=mp->m_descriptor;

            int bestDist = 256;
            int secondBestDist=256;
            int bestIdx2 = -1;

            for (int j = 0; j <KP.size() ; ++j)
            {
                cv::circle(CurrentFrame->detectimg,KP[j].pt,5,cv::Scalar(0,0,256),-1);
                mappoint *mp_j=CurrentFrame->MapPoints[j];
                if(mp_j)
                    continue;

                const cv::Mat &d = CurrentFrame->f_descriptor.row(j);
                const int dist = DescriptorDistance(dMP,d);

                if(dist<bestDist)
                {
                    secondBestDist=bestDist;
                    bestDist=dist;
                    bestIdx2=j;
                }
            }
            cv::Point2f cur=CurrentFrame->keypoints_l[bestIdx2].pt;
            cv::Point2f last=LastFrame.keypoints_l[i].pt;
            cv::Point2f k_point=cv::Point2f(cur.x,cur.y+CurrentFrame->leftimg.rows)-last;
            CurrentFrame->match_score[i]=(float(secondBestDist)/float(bestDist));

            float gradient=0;
            bool pp1;
            bool pp3;
            pp1=bestDist<30;
            pp3=bestDist<55&&CurrentFrame->match_score[i]>1.25;// 最好参数

            if(pp1||pp3)
            {
                good++;
                if(k_point.x!=0)
                  gradient=k_point.y/k_point.x;
                 else
                    gradient=99999;

                if((gradient<0&&gradient>-2)||gradient>0&&gradient<2)
                {
                    cur_gradient.push_back(gradient);
                    k_bad++;
                    continue;
                }

                for (int k = 0; k < CurrentFrame->offline_box.size(); ++k) {
                    int left=CurrentFrame->offline_box[k][0];
                    int right=CurrentFrame->offline_box[k][1];
                    int top=CurrentFrame->offline_box[k][2];
                    int bottom=CurrentFrame->offline_box[k][3];
                    if(cur.x>left-5&&cur.x<right+5&&cur.y>top-5&&cur.y<bottom+5)
                    {
                        dynamic=true;
                        break;
                    }
                }

                //// disable following lines if using offline semantic ////
//                for (int k = 0; k < CurrentFrame->boxes.size(); ++k) {
//                    int x0=CurrentFrame->boxes[k].tl().x;
//                    int y0=CurrentFrame->boxes[k].tl().y;
//                    int x1=CurrentFrame->boxes[k].br().x;
//                    int y1=CurrentFrame->boxes[k].br().y;
//                    if(cur.x>x0&&cur.x<x1&&cur.y>y0&&cur.y<y1)
//                    {
//                        if(CurrentFrame->id<=1)
//                            CurrentFrame->DY_keypoints.push_back(cur);
//                        dynamic=true;
//                        break;
//                    }
//                }

                if(dynamic)
                {
                    mp->bad=true;
                    class_bad++;
                    continue;
                }
                else
                {
                    matchess++;
                    last_po.push_back(last);
                    cur_po.push_back(cur);
                    cur_score.push_back(bestDist);
                    CurrentFrame->MapPoints[bestIdx2]=mp;
                    mp->AddObservation(CurrentFrame,bestIdx2);
                }
            }
        }
    }

    cout<<"匹配点:"<<good<<"~~"<<matchess<<endl;
    int h=LastFrame.detectimg.rows;
    int w=LastFrame.detectimg.cols;

    LastFrame.detectimg.copyTo(outImg.rowRange(0,h));
    CurrentFrame->detectimg.copyTo(outImg.rowRange(h+1,outImg.rows));

    for (int kk = 0; kk <cur_po.size() ; ++kk) {
            cv::line(outImg,last_po[kk],cv::Point2f(cur_po[kk].x,CurrentFrame->detectimg.rows+cur_po[kk].y),cv::Scalar(0,256,0));///bgr

    }
//    for (int pp = 0; pp <cur_pro.size() ; ++pp) {
//        cv::line(outImg,last_fea[pp],cv::Point2f(cur_pro[pp].x,CurrentFrame->detectimg.rows+cur_pro[pp].y),cv::Scalar(256,0,0));///bgr
//
//    }

    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    vector<int> pts2d_index;

    for (int j = 0; j <CurrentFrame->N ; ++j) {
        mappoint *mapp=CurrentFrame->MapPoints[j];
        if(mapp)
        {
            pts2d_index.push_back(j);
            pts2d.push_back ( CurrentFrame->keypoints_l[j].pt );
            pts3d.push_back( cv::Point3f( mapp->worldpos.at<float>(0,0), mapp->worldpos.at<float>(1,0), mapp->worldpos.at<float>(2,0)));
        }
    }

    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers );////double

    int num_inliers_ = inliers.rows;
    inlier_ratio=(float)num_inliers_/(float)pts2d.size();
    cout<<"内点比例:"<<inlier_ratio<<endl;
//    if(CurrentFrame->id>595)
//    {
        cv::imshow("result",outImg);
        cv::waitKey(0);
//    }

    cv::Mat Cur_Rcw(3,3,CV_32F);
    cv::Rodrigues(rvec, Cur_Rcw);

    cv::Mat Tcl;
    Tcl=(cv::Mat_<float>(4,4)<<
                             Cur_Rcw.at<double>(0,0),Cur_Rcw.at<double>(0,1),Cur_Rcw.at<double>(0,2),tvec.at<double>(0,0),
            Cur_Rcw.at<double>(1,0),Cur_Rcw.at<double>(1,1),Cur_Rcw.at<double>(1,2),tvec.at<double>(1,0),
            Cur_Rcw.at<double>(2,0),Cur_Rcw.at<double>(2,1),Cur_Rcw.at<double>(2,2),tvec.at<double>(2,0),
            0,0,0,1);

    CurrentFrame->SetPose(Tcl*CurrentFrame->Tcw);

    return inlier_ratio;

}

int pnpmatch::poseEstimationlocalmap(frame *CurrentFrame,set<mappoint*> &localmappoints,cv::Mat &K)
{
    for (set<mappoint*>::iterator it = localmappoints.begin(); it != localmappoints.end(); ++it)
    {
        mappoint *mp_2=*it;
        if(mp_2)
        {
            if(CurrentFrame->id-mp_2->create_id-6>mp_2->observation_num)
            {
                localmappoints.erase(mp_2);
                continue;
            }

            vector<cv::KeyPoint> KP=CurrentFrame->keypoints_l;
            cv::Mat dMP=mp_2->m_descriptor;//用上一针描述子比较准
            int bestDist = 256;

            int secondBestDist=256;
            int bestIdx2 = -1;

            for (int j = 0; j <KP.size() ; ++j) {
                mappoint *mp_j=CurrentFrame->MapPoints[j];
                if(mp_j&&!mp_j->bad)
                    continue;

                const cv::Mat &d = CurrentFrame->f_descriptor.row(j);
                const int dist = DescriptorDistance(dMP,d);
                if(dist<bestDist)
                {
                    secondBestDist=bestDist;
                    bestDist=dist;
                    bestIdx2=j;
                }
            }

            bool pp1=bestDist<20;
            bool pp3=bestDist<30&&(float)secondBestDist/(float)bestDist>1.5;// 最好参数
            if(pp1||pp3)
            {
                CurrentFrame->MapPoints[bestIdx2]=mp_2;
                mp_2->AddObservation(CurrentFrame,bestIdx2);
//                cout<<mp_2->create_id<<"  "<<mp_2->observation_num<<"  "<<bestDist<<endl;
            }
        }
    }

    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    vector<int> pts2d_index;

    for (int j = 0; j <CurrentFrame->N ; ++j) {
        mappoint *mapp=CurrentFrame->MapPoints[j];
        if(mapp)
        {
            pts2d_index.push_back(j);
            pts2d.push_back ( CurrentFrame->keypoints_l[j].pt );
            pts3d.push_back( cv::Point3f( mapp->worldpos.at<float>(0,0), mapp->worldpos.at<float>(1,0), mapp->worldpos.at<float>(2,0)));
        }
    }

    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers );////double


    cv::Mat Cur_Rcw(3,3,CV_32F);
    cv::Rodrigues(rvec, Cur_Rcw);

    cv::Mat Tcl;
    Tcl=(cv::Mat_<float>(4,4)<<
                             Cur_Rcw.at<double>(0,0),Cur_Rcw.at<double>(0,1),Cur_Rcw.at<double>(0,2),tvec.at<double>(0,0),
            Cur_Rcw.at<double>(1,0),Cur_Rcw.at<double>(1,1),Cur_Rcw.at<double>(1,2),tvec.at<double>(1,0),
            Cur_Rcw.at<double>(2,0),Cur_Rcw.at<double>(2,1),Cur_Rcw.at<double>(2,2),tvec.at<double>(2,0),
            0,0,0,1);

    CurrentFrame->SetPose(Tcl*CurrentFrame->Tcw);
}
