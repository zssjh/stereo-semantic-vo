//
// Created by zss on 18-12-17.
//

#include <convert.h>
g2o::SE3Quat convert::toSE3Quat(const cv::Mat &cvT)
{
    Eigen::Matrix<double,3,3> R;
    R << cvT.at<float>(0,0), cvT.at<float>(0,1), cvT.at<float>(0,2),
            cvT.at<float>(1,0), cvT.at<float>(1,1), cvT.at<float>(1,2),
            cvT.at<float>(2,0), cvT.at<float>(2,1), cvT.at<float>(2,2);

    Eigen::Matrix<double,3,1> t(cvT.at<float>(0,3), cvT.at<float>(1,3), cvT.at<float>(2,3));

    return g2o::SE3Quat(R,t);
}

cv::Mat convert::R_t_to_Twc(const cv::Mat &Rcw,const cv::Mat &tvec)
{
    cv::Mat Rwc;
    Rwc=(cv::Mat_<float>(3,3)<<
            Rcw.at<float>(0,0),Rcw.at<float>(1,0),Rcw.at<float>(2,0),
            Rcw.at<float>(0,1),Rcw.at<float>(1,1),Rcw.at<float>(2,1),
            Rcw.at<float>(0,2),Rcw.at<float>(1,2),Rcw.at<float>(2,2));

    cv::Mat twc;
    twc=-Rwc*tvec;

    cv::Mat Twc;
    Twc= (cv::Mat_<float>(4,4) <<
                               Rwc.at<float>(0,0),Rwc.at<float>(0,1),Rwc.at<float>(0,2),twc.at<float>(0,0),
            Rwc.at<float>(1,0),Rwc.at<float>(1,1),Rwc.at<float>(1,2),twc.at<float>(1,0),
            Rwc.at<float>(2,0),Rwc.at<float>(2,1),Rwc.at<float>(2,2),twc.at<float>(2,0),
            0,0,0,1);
    return Twc;
}

cv::Mat convert::R_t_to_Tcw(const cv::Mat &Rcw,const cv::Mat &tvec)
{
    cv::Mat Tcw;
    Tcw= (cv::Mat_<float>(4,4) <<
            Rcw.at<float>(0,0),Rcw.at<float>(0,1),Rcw.at<float>(0,2),tvec.at<float>(0,0),
            Rcw.at<float>(1,0),Rcw.at<float>(1,1),Rcw.at<float>(1,2),tvec.at<float>(1,0),
            Rcw.at<float>(2,0),Rcw.at<float>(2,1),Rcw.at<float>(2,2),tvec.at<float>(2,0),
            0,0,0,1);
    return Tcw;
}

cv::Mat convert::toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat convert::toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

Eigen::Matrix<double,3,3> convert::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
            cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
            cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> convert::toQuaternion(const cv::Mat &M)
{
    Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = q.x();
    v[1] = q.y();
    v[2] = q.z();
    v[3] = q.w();

    return v;
}

