#include <cstdio>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "MSA.h"
using namespace std;
using namespace cv;

MSA solver;

int main()
{

//    Mat left = imread("/home/zss/CLionProjects/MB/img/ArtL0.png"), right = imread("/home/zss/CLionProjects/MB/img/ArtL1.png");
    //Mat left = imread("tsukuba0.ppm"), right = imread("tsukuba1.ppm");//ndisp = 15 * 16;
    //Mat left = imread("Venus0.ppm"), right = imread("Venus1.ppm");//ndisp = 19 * 8;
    Mat left = imread("/home/zss/CLionProjects/MB/img/Teddy0.png"), right = imread("/home/zss/CLionProjects/MB/img/Teddy1.png");//ndisp = 59 * 4;
    //Mat left = imread("Cones0.png"), right = imread("Cones1.png");//ndisp = 59 * 4;
    //Mat left = imread("Adirondack0.png"), right = imread("Adirondack1.png");
    //Mat left = imread("Motorcycle0.png"), right = imread("Motorcycle1.png");//ndisp = 260/4=67 * 3;
    //Mat left = imread("Piano0.png"), right = imread("Piano1.png");//ndisp = 196/4=49 * 5;
    //Mat left = imread("Pipes0.png"), right = imread("Pipes1.png");
    //Mat left = imread("Playroom0.png"), right = imread("Playroom1.png");
    //Mat left = imread("Recycle0.png"), right = imread("Recycle1.png");//72
    //Mat left = imread("Shelves0.png"), right = imread("Shelves1.png");//47
    //Mat left = imread("PianoL0.png"), right = imread("PianoL1.png");//53
    //Mat left = imread("MotorcycleE0.png"), right = imread("MotorcycleE1.png");//53
    //Mat left = imread("Jadeplant0.png"), right = imread("Jadeplant1.png");//78
    //Mat left = imread("bowling21.png"), right = imread("bowling25.png");//68
    //Mat left = imread("cloth11.png"), right = imread("cloth15.png");//59
    //Mat left = imread("cloth21.png"), right = imread("cloth25.png");//78
    //Mat left = imread("cloth31.png"), right = imread("cloth35.png");//57
    //Mat left = imread("cloth41.png"), right = imread("cloth45.png");//69
    //Mat left = imread("Flowerpots1.png"), right = imread("Flowerpots5.png");//62
    //Mat left = imread("Lampshade11.png"), right = imread("Lampshade15.png");//66
    //Mat left = imread("Lamp1.png"), right = imread("Lamp5.png");//67
    //Mat left = imread("Midd11.png"), right = imread("Midd15.png");//70
    //Mat left = imread("Mid1.png"), right = imread("Mid5.png");//63
    //Mat left = imread("Monopoly1.png"), right = imread("Monopoly5.png");//55
    //Mat left = imread("Plastic1.png"), right = imread("Plastic5.png");//67
    //Mat left = imread("rocks11.png"), right = imread("rocks15.png");//58
    //Mat left = imread("rocks1.png"), right = imread("rocks5.png");//58
    //Mat left = imread("wood11.png"), right = imread("wood15.png");//78
    //Mat left = imread("woods1.png"), right = imread("woods5.png");//74
    //Mat left = imread("Art01.png"), right = imread("Art05.png");//76
    //Mat left = imread("books1.png"), right = imread("books5.png");//75
    //Mat left = imread("dolls1.png"), right = imread("dolls5.png");//75
    //Mat left = imread("Laundry1.png"), right = imread("Laundry5.png");//76
    //Mat left = imread("Moebius1.png"), right = imread("Moebius5.png");//74
    //Mat left = imread("Reindeer1.png"), right = imread("Reindeer5.png");//69

    if (!left.data || !right.data) {
        printf("can't read the image\n");
        return 0;
    }

    int d = 59, scale = 4;

    solver.solve(left, right, d, scale, true);

//    waitKey();

    return 0;
}
