//
// Created by zss on 18-12-17.
//

#include <pnpmatch.h>
#include <set>
#include <opencv/cv.hpp>
#include <convert.h>
#include <opencv/cxeigen.hpp>
#include <Tracking.h>
using namespace cv;

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
    cv::Mat fundamental_matrix;
    pnpmatch::poseEstimation2D_2D(CurrentFrame,LastFrame,K,fundamental_matrix);



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
    int class_bad=0;
    int matches=0;

    for (int i = 0; i <LastFrame.N ; ++i) {
        mappoint *mp=LastFrame.MapPoints[i];

        if(mp&&!mp->bad)
        {
            cv::circle(LastFrame.detectimg,LastFrame.keypoints_l[i].pt,5,cv::Scalar(0,0,256),-1);
            cv::Mat Pw=mp->worldpos;

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
            cv::Point2f cur=CurrentFrame->keypoints_l[bestIdx2].pt;//////极线约束
            cv::Point2f last=LastFrame.keypoints_l[i].pt;//////极线约束

            CurrentFrame->match_score[i]=(float(secondBestDist)/float(bestDist));

            if(bestDist<15)
            {
                for (int k = 0; k < CurrentFrame->offline_box.size(); ++k) {
                    int left=CurrentFrame->offline_box[k][0];
                    int right=CurrentFrame->offline_box[k][1];
                    int top=CurrentFrame->offline_box[k][2];
                    int bottom=CurrentFrame->offline_box[k][3];
                    if(cur.x>left-10&&cur.x<right+10&&cur.y>top-10&&cur.y<bottom+10)
                    {
                        double A = fundamental_matrix.at<double>(0, 0)*last.x + fundamental_matrix.at<double>(0, 1)*last.y + fundamental_matrix.at<double>(0, 2);
                        double B = fundamental_matrix.at<double>(1, 0)*last.x + fundamental_matrix.at<double>(1, 1)*last.y + fundamental_matrix.at<double>(1, 2);
                        double C = fundamental_matrix.at<double>(2, 0)*last.x + fundamental_matrix.at<double>(2, 1)*last.y + fundamental_matrix.at<double>(2, 2);
                        double dd = fabs(A*cur.x + B*cur.y + C) / sqrt(A*A + B*B); //Epipolar constraints
                        cout<<dd<<endl;
                        if (dd > 0.1)
                        {
                            dynamic=true;
                            break;
                        }

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
                    matches++;
                    last_po.push_back(last);
                    cur_po.push_back(cur);
                    cur_score.push_back(bestDist);
                    CurrentFrame->MapPoints[bestIdx2]=mp;
                    mp->AddObservation(CurrentFrame,bestIdx2);
                }
            }
        }
    }


    int nummm=0;
    for (set<mappoint*>::iterator it2 = localmappoints.begin(); it2 != localmappoints.end(); it2++)
    {
        mappoint *mp_local=*it2;
        if(mp_local&&!mp_local->bad)
        {
            if(mp_local->observations.count(CurrentFrame))
            {

                continue;//// 去掉刚匹配过的点,局部地图点删除导致从第6帧开始和matches数量不一致
            }

            vector<cv::KeyPoint> KP=CurrentFrame->keypoints_l;
            cv::Mat dMP=mp_local->m_descriptor;
            int bestDist = 256;
            int secondBestDist=256;
            int bestIdx2 = -1;

            for (int j = 0; j <KP.size() ; ++j) {
                mappoint *mp_j=CurrentFrame->MapPoints[j];
                if(mp_j)
                    continue;//// 已经匹配的有很多错误的---597帧

                const cv::Mat &d = CurrentFrame->f_descriptor.row(j);
                const int dist = DescriptorDistance(dMP,d);
                if(dist<bestDist)
                {
                    secondBestDist=bestDist;
                    bestDist=dist;
                    bestIdx2=j;
                }
            }

            if(bestDist<30&&(float)secondBestDist/(float)bestDist>2)
            {
                nummm++;
                CurrentFrame->MapPoints[bestIdx2]=mp_local;
                mp_local->AddObservation(CurrentFrame,bestIdx2);
            }
        }
    }

    int h=LastFrame.detectimg.rows;
    int w=LastFrame.detectimg.cols;

    LastFrame.detectimg.copyTo(outImg.rowRange(0,h));
    CurrentFrame->detectimg.copyTo(outImg.rowRange(h+1,outImg.rows));

    for (int kk = 0; kk <cur_po.size() ; ++kk) {
            cv::line(outImg,last_po[kk],cv::Point2f(cur_po[kk].x,CurrentFrame->detectimg.rows+cur_po[kk].y),cv::Scalar(0,256,0));///bgr

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

    int num_inliers_ = inliers.rows;
    inlier_ratio=(float)num_inliers_/(float)pts2d.size();
//    if(CurrentFrame->id>595)
//    {
        cv::imshow("result",outImg);
        cv::waitKey(100);
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

void pnpmatch::find_feature_matches ( const cv::Mat& img_1, const cv::Mat& img_2,
                                   std::vector<cv::KeyPoint>& keypoints_1,
                                   std::vector<cv::KeyPoint>& keypoints_2,
                                   std::vector<cv::DMatch >& matches )
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<cv::DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }


    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

int pnpmatch::poseEstimation2D_2D(frame *CurrentFrame,frame &LastFrame,cv::Mat &K,cv::Mat &fundamental_matrix)
{
   int box_size=CurrentFrame->offline_box.size();
    vector<cv::DMatch> matches;
    find_feature_matches(CurrentFrame->leftimg,LastFrame.leftimg,CurrentFrame->keypoints_l,LastFrame.keypoints_l,matches);

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1_background;
    vector<Point2f> points2_background;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        bool dynamic=false;
        Point2f cur  = CurrentFrame->keypoints_l[matches[i].queryIdx].pt;
        Point2f last = LastFrame.keypoints_l[matches[i].trainIdx].pt;

        for (int k = 0; k <box_size ; ++k) {
            int left=CurrentFrame->offline_box[k][0];
            int right=CurrentFrame->offline_box[k][1];
            int top=CurrentFrame->offline_box[k][2];
            int bottom=CurrentFrame->offline_box[k][3];
            if(cur.x>left-10&&cur.x<right+10&&cur.y>top-10&&cur.y<bottom+10)
            {
                dynamic=true;
                break;
            }
        }
        if(!dynamic)
        {
            points1_background.push_back (cur);
            points2_background.push_back (last);
        }
    }

    fundamental_matrix = findFundamentalMat ( points1_background, points2_background, CV_FM_8POINT );
}

