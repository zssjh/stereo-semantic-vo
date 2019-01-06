//
// Created by zss on 18-12-15.
//
#include <frame.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  ////包含cv::imshow
#include <assert.h>
#include <Thirdparty/libelas/src/elas.h>
#include <Thirdparty/libelas/src/image.h>
#include <opencv2/calib3d.hpp>
#include <chrono>

using namespace cv;

frame::frame()
{

}

frame::frame( frame *mframe):K(mframe->K.clone()),id(mframe->id),f_descriptor(mframe->f_descriptor.clone()),bf(mframe->bf),width(mframe->width),
                                  height(mframe->height),keypoints_l(mframe->keypoints_l),N(mframe->N), MapPoints(mframe->MapPoints),
                             depthimg(mframe->depthimg.clone()),dispimg(mframe->dispimg.clone()),leftimg(mframe->leftimg), rightimg(mframe->rightimg),
                             fx(mframe->fx),fy(mframe->fy),cx(mframe->cx),cy(mframe->cy),inlier(mframe->inlier),match_score(mframe->match_score),timestamp(mframe->timestamp),
                             keypoints_r(mframe->keypoints_r),boxes(mframe->boxes),have_detected(mframe->have_detected)
{

    if(!mframe->Tcw.empty())
        SetPose(mframe->Tcw);
}

frame::frame( cv::Mat &imLeft,  cv::Mat &imRight, cv::Mat &imdepth,double &time_tamp,cv::Mat &K_,float &mbf)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    K=K_;
    fx=K.at<float>(0,0);
    fy=K.at<float>(1,1);
    cx=K.at<float>(0,2);
    cy=K.at<float>(1,2);
    boxes.reserve(10);
    have_detected= false;
    timestamp=time_tamp;
    bf=mbf;
    leftimg=imLeft.clone();
    rightimg=imRight.clone();
    width=imLeft.cols;
    height=imLeft.rows;
    SetPose(cv::Mat::eye(4,4,CV_32F));


    featuredetect(imLeft);


    N = keypoints_l.size();
    keypoints_r.reserve(N);
    MapPoints = vector<mappoint*>(N,static_cast<mappoint*>(NULL));
    inlier = vector<bool>(N,false);
    match_score=vector<float>(N,-1);
    depthimg = cv::Mat(height,width,CV_32F,-1);
    dispimg = cv::Mat(height,width,CV_32F,-1);

    dispimg=ElasMatch(imLeft,imRight);//// SGBM计算的CV_16s，转化为CV_8U
    computekeypoint_r();

    disp2Depth(bf);
}

void frame::SetPose( cv::Mat mTcw)
{
    Tcw = mTcw.clone();
    Rcw = Tcw.rowRange(0,3).colRange(0,3);
    Rwc = Rcw.t();
    tcw = Tcw.rowRange(0,3).col(3);
    twc = -Rwc*tcw;
}

void frame::featuredetect(cv::Mat &img)
{
    Ptr<ORB> orb = ORB::create();
    orb->detectAndCompute(img, Mat(), keypoints_l, f_descriptor);
}


 cv::Mat frame::ElasMatch( cv::Mat &leftImage,cv::Mat &rightImage)
{
    int numberOfDisparities = ((leftImage.rows / 8) + 15) & -16;////48 必须是16的倍数
    cv::Mat disp;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
    sgbm->setPreFilterCap(63);
    int SADWindowSize = 9;
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = leftImage.channels();
    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    sgbm->compute(leftImage, rightImage, disp);

    Mat disp8(disp.size(),CV_32F);
    disp.convertTo(disp8, CV_32F, 1.0/16);//-16-----> -1

    return disp8;
}

void frame::computekeypoint_r()
{
    float lx=0;
    float ly=0;
    float rx=-1;
    float ry=0;
    float disparity=0;
    for (int i = 0; i <keypoints_l.size() ; ++i) {
        lx=keypoints_l[i].pt.x;
        ly=keypoints_l[i].pt.y;
        disparity=dispimg.at<float>(ly,lx);
        if(disparity!=-1)
            rx=lx-disparity;
        ry=ly;
        keypoints_r.push_back(cv::Point2f(rx,ry));
    }
}

cv::Mat frame::ElasMatch2( cv::Mat &leftImage,cv::Mat &rightImage)
{
    cv::Mat disp_l,disp_r,disp8u_l,disp8u_r;
    double minVal; double maxVal; //视差图的极值

    // 计算视差
    // generate disparity image using LIBELAS

    int bd = 0;
    const int32_t dims[3] = {leftImage.cols,leftImage.rows,leftImage.cols};
    cv::Mat leftdpf = cv::Mat::zeros(cv::Size(leftImage.cols,leftImage.rows), CV_32F);
    cv::Mat rightdpf = cv::Mat::zeros(cv::Size(leftImage.cols,leftImage.rows), CV_32F);
    Elas::parameters param;
    param.postprocess_only_left = false;
    Elas elas(param);
    elas.process(leftImage.data,rightImage.data,leftdpf.ptr<float>(0),rightdpf.ptr<float>(0),dims);

    cv::Mat(leftdpf(cv::Rect(bd,0,leftImage.cols,leftImage.rows))).copyTo(disp_l);
    cv::Mat(rightdpf(cv::Rect(bd,0,rightImage.cols,rightImage.rows))).copyTo(disp_r);
//
//    //-- Check its extreme values
//    cv::minMaxLoc( disp_l, &minVal, &maxVal );
//    cout<<"范围："<<maxVal-minVal<<endl;
//
//    //-- Display it as a CV_8UC1 image
//    cout<<"type:"<<disp_l.type()<<endl;
//    disp_l.convertTo(disp8u_l, CV_8U, 255/(maxVal - minVal));//(numberOfDisparities*16.)
//
//    cv::minMaxLoc( disp_r, &minVal, &maxVal );
//
//    //-- Display it as a CV_8UC1 image
//    disp_r.convertTo(disp8u_r, CV_8U, 255/(maxVal - minVal));//(numberOfDisparities*16.)
//
//    cv::normalize(disp8u_l, disp8u_l, 0, 255, CV_MINMAX, CV_8UC1);    // obtain normalized image
//    cv::normalize(disp8u_r, disp8u_r, 0, 255, CV_MINMAX, CV_8UC1);    // obtain normalized image
    return disp_r;

}

void frame::disp2Depth( float bf)
{
    cv::Mat dispMap=dispimg;
    int type = dispMap.type();
    int height = dispMap.rows;
    int width = dispMap.cols;

    cv::Mat depthMap(height, width, CV_32F, cv::Scalar(-1));
    float* dispData = (float*)dispMap.data;
    float * depthData = (float*)depthMap.data;

    int boxsize=9;
    int half_boxsize=boxsize/2;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int id = i*width + j;
            if (!dispData[id])
                continue;
//            if(dispData[id]==-1&&i>=half_boxsize&&i<=height-half_boxsize&&j>=half_boxsize&&j<=width-half_boxsize)
//            {
//                float sum=0;int count=0;
//                cout<<"[";
//                for (int k = i-half_boxsize; k <=i+half_boxsize ; ++k) {
//                    for (int l = j-half_boxsize; l <=j+half_boxsize ; ++l) {
//                        float hole=dispMap.at<float>(k,l);
//                        cout<<hole<<" ";
//                        if(hole!=-1)
//                        {
//                            count++;
//                            sum+=hole;
//                        }
//                    }
//                }
//                cout<<endl;
//                if(count!=0)
//                {
//                    float hole_disp=sum/(float)count;
//                    dispData[id]=hole_disp;
//                    cout<<hole_disp<<endl;
//                }
//
//            }
            depthData[id] = bf / dispData[id];
        }
    }
//    insertDepth32f(depthMap);
    depthimg=depthMap;
//    cv::waitKey(0);

//    cv::imshow("disp2",depthMap);
}

cv::Mat frame::UnprojectStereo(const float &u,const float &v,const float &z)
{
    cv::Mat x3Dc;
    cv::Mat x3D;//(3,1,CV_32F,cv::Scalar(1))
    if(z>0)
    {
        const float x = (u-cx)*z*(1/fx);
        const float y = (v-cy)*z*(1/fy);
        x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        x3D=Rwc*x3Dc+twc;
        return x3D;
    }
    else
        return cv::Mat();
}

void frame::createmappoint(set<mappoint*> &localmap)
{
    int num=0;
    for (int i = 0; i < N; ++i) {
        mappoint *mp=MapPoints[i];
        if (!mp)
        {
            num++;
//            cout<<"creatmappoint"<<endl;
            const float u = keypoints_l[i].pt.x;
            const float v = keypoints_l[i].pt.y;
            const float z = depthimg.at<float>(v,u);
            if (z > 0) {
                cv::Mat x3D = UnprojectStereo(u,v,z);
                mappoint *newmp = new mappoint(x3D,this,i);////什么时候定义*,*的时候是->
                newmp->AddObservation(this,i);
                newmp->create_id=id;
                MapPoints[i] = newmp;
//                localmap.insert(newmp);
            }
        }
    }
//    cout<<"新建匹配点:"<<num<<endl;
}


