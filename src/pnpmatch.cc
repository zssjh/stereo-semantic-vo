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


int pnpmatch::poseEstimationPnP(frame *CurrentFrame,frame &LastFrame,cv::Mat &K)
{
    float inlier_ratio=0;
    int matches=0;

    cv::Mat outImg(CurrentFrame->leftimg.rows+LastFrame.leftimg.rows+1,CurrentFrame->leftimg.cols,CurrentFrame->leftimg.type());
    vector<cv::Point2f> cur_po;vector<int > cur_score;vector<cv::Point2f> last_po;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (int i = 0; i <LastFrame.N ; ++i) {

      cv::circle(LastFrame.leftimg,LastFrame.keypoints_l[i].pt,5,cv::Scalar(0,0,256),-1);
        mappoint *mp=LastFrame.MapPoints[i];
        if(mp)
        {
            vector<cv::KeyPoint> KP=CurrentFrame->keypoints_l;
            vector<cv::Point2f> KP_r=CurrentFrame->keypoints_r;

            cv::Mat dMP=mp->m_descriptor;

            int bestDist = 256;
            int secondBestDist=256;
            int bestIdx2 = -1;

            for (int j = 0; j <KP.size() ; ++j)
            {
                bool dynamic=false;
                cv::circle(CurrentFrame->leftimg,KP[j].pt,5,cv::Scalar(0,0,256),-1);
                mappoint *mp_j=CurrentFrame->MapPoints[j];
                if(mp_j)
                    continue;

                for (int k = 0; k < CurrentFrame->boxes.size(); ++k) {

                    int x0=CurrentFrame->boxes[k].tl().x;
                    int y0=CurrentFrame->boxes[k].tl().y;
                    int x1=CurrentFrame->boxes[k].br().x;
                    int y1=CurrentFrame->boxes[k].br().y;
                    if(KP[j].pt.x>x0&&KP[j].pt.x<x1&&KP[j].pt.y>y0&&KP[j].pt.y<y1)
                    {
                        dynamic=true;
//                        cout<<"dynamic"<<endl;
                        break;
                    }
                }

                if(dynamic)
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
            CurrentFrame->match_score[i]=(float(secondBestDist)/float(bestDist));

            bool pp1=bestDist<30;
            bool pp3=bestDist<55&&CurrentFrame->match_score[i]>1.25;// 最好参数
            if(pp1||pp3)
            {
                cur_po.push_back(cur);
                cur_score.push_back(bestDist);
                last_po.push_back(last);
                CurrentFrame->MapPoints[bestIdx2]=mp;
                matches++;
                mp->AddObservation(CurrentFrame,bestIdx2);
            }
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    int h=LastFrame.leftimg.rows;
    int w=LastFrame.leftimg.cols;

    LastFrame.leftimg.copyTo(outImg.rowRange(0,h));
    CurrentFrame->leftimg.copyTo(outImg.rowRange(h+1,outImg.rows));
//
    for (int k = 0; k <cur_po.size() ; ++k) {
//        if(cur_score[k]>50)
//        {
//            cv::putText(outImg,to_string(cur_score[k]),last_po[k],cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,0,0),1,8);
//            cv::putText(outImg,to_string(cur_score[k]),last_po[k],cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,0,0),1,8);
            cv::line(outImg,last_po[k],cv::Point2f(cur_po[k].x,CurrentFrame->leftimg.rows+cur_po[k].y),cv::Scalar(0,256,0));
//        if(CurrentFrame->id==467)
//            cout<<cur_po[k]<<endl;
//        }

    }
//    cv::imshow("result",outImg);
//    cv::waitKey(100);

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
}
