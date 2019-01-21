//
// Created by zss on 18-12-20.
//
#include <view.h>
#include <opencv/cv.hpp>

void View::DrawGraph(frame &lastframe,frame *currentframe)
{
    glLineWidth(2);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);

    cv::Mat Ow = lastframe.twc;
    cv::Mat Ow2=currentframe->twc;
    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
    glEnd();
}

void View::DrawMappoints(set<mappoint*> &spRefMPs,int id)
{
    cout<<spRefMPs.size()<<endl;
    glPointSize(2);
    glBegin(GL_POINTS);

    glColor3f(1 - (float) id / 1101, 0.0f, (float) id / 1101);

    for(set<mappoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {

        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();
//    pangolin::FinishFrame();
}

