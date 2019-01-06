
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/affine.hpp>

#include <iostream>
#include <vector>
#include<iomanip>
#include <fstream>
#include <Tracking.h>
#include <pangolin/pangolin.h>
#include <include/view.h>
#include <unistd.h>
#include <thread>
#include <include/semantic.h>


using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<string> &vstrImagedepth,vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_2/";
    string strPrefixRight = strPathToSequence + "/image_3/";
    string strPrefixdepth = strPathToSequence + "/01/images/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);
    vstrImagedepth.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
        vstrImagedepth[i] = strPrefixdepth + ss.str() + ".png";
    }
}


int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<string> vstrImagedepth;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight,vstrImagedepth, vTimestamps);
    const int nImages = vstrImageLeft.size();

    Tracking* mpTracker;
    mpTracker = new Tracking(argv[2]);

    pangolin::CreateWindowAndBind("stereo_vo: Map Viewer",1200,768);
    glEnable(GL_DEPTH_TEST);
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
//
//    // 添加显示机器人位置选择项
//    pangolin::Var<float> menuDisplayPose_x("menu.Pose:X",false,false);
//    pangolin::Var<float> menuDisplayPose_Y("menu.Pose:Y",false,false);
//    pangolin::Var<float> menuDisplayPose_Z("menu.Pose:Z",false,false);
//
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1200,768,500,500,10,10,0.01,1000),
            pangolin::ModelViewLookAt(0,-900,-10, 0,0,0,0.0,-1.0, 0.0)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc_M;
    Twc_M.SetIdentity();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    ofstream f;
    f.open("cameratrajectory_kitti.txt");
    f << fixed;

    ofstream f2;
    f2.open("cameratrajectory_tum.txt");
    f2 << fixed;

    Semantic* mpSemanticer = new Semantic(vstrImageLeft,vstrImageRight);
    std::thread* mpsemantic = new thread(&Semantic::Run,mpSemanticer);

    mpSemanticer->SetTracker(mpTracker);
    mpTracker->Setsemanticer(mpSemanticer);


    cv::Mat imLeft, imRight,imdepth;
    for(int ni=0; ni<nImages; ni++) {
        imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imLeft.empty()) {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        cout << "load image " << ni << endl;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        //// static函数不用定义对象实体
        mpTracker->Track(imLeft, imRight,imdepth, tframe,mpTracker->K,mpTracker->bf,f,f2,Twc_M);

//        s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-100,-0.1, 0,0,0,0.0,-1.0, 0.0));
        s_cam.Follow(Twc_M);
        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,1.0f);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }
    f.close();
    f2.close();
    cout << endl << "trajectory saved!" << endl;

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    return 0;
}

