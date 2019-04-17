#include <iostream>
#include <fstream>
#include <ctime>
#include <queue>
#include <string>
#include <cv.h>
#include <unistd.h>
#include <highgui.h>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/ocl.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <string.h>
#include <sys/time.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

const CvScalar COLOR_BLUE = CvScalar(255, 0, 0);
const CvScalar COLOR_RED = CvScalar(0, 0, 255);
const CvScalar COLOR_GREEN = CvScalar(0, 255, 0);

const Vec3b RGB_WHITE_LOWER = Vec3b(100, 100, 190);
const Vec3b RGB_WHITE_UPPER = Vec3b(255, 255, 255);
const Vec3b RGB_YELLOW_LOWER = Vec3b(225, 180, 0);
const Vec3b RGB_YELLOW_UPPER = Vec3b(255, 255, 170);
const Vec3b HSV_YELLOW_LOWER = Vec3b(0, 120, 80);
const Vec3b HSV_YELLOW_UPPER = Vec3b(40, 255, 255);

const Vec3b HLS_YELLOW_LOWER = Vec3b(20, 120, 80);
const Vec3b HLS_YELLOW_UPPER = Vec3b(45, 200, 255);

const Vec3b HSV_BLUE_LOWER = Vec3b(80, 200, 65);
const Vec3b HSV_BLUE_UPPER = Vec3b(120, 255, 180);

const Vec3b HSV_RED_LOWER = Vec3b(0, 100, 100);
const Vec3b HSV_RED_UPPER = Vec3b(10, 255, 255);
const Vec3b HSV_RED_LOWER1 = Vec3b(160, 100, 100);
const Vec3b HSV_RED_UPPER1 = Vec3b(179, 255, 255);
int main (int argc, const char * argv[])
{
    VideoCapture cap(1);


    if (!cap.isOpened())
        return -1;

    Mat img;


    while (true)
    {
        cap >> img;

        cvtColor(img, img, CV_RGB2GRAY);

        resize( img, img, Size(64, 128), 0, 0, CV_INTER_NN );



HOGDescriptor d;
// Size(128,64), //winSize
// Size(16,16), //blocksize
// Size(8,8), //blockStride,
// Size(8,8), //cellSize,
// 9, //nbins,
// 0, //derivAper,
// -1, //winSigma,
// 0, //histogramNormType,
// 0.2, //L2HysThresh,
// 0 //gammal correction,
// //nlevels=64
//);

// void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
//                             Size winStride, Size padding,
//                             const vector<Point>& locations) const
vector<float> descriptorsValues;
vector<Point> locations;
d.compute( img, descriptorsValues, Size(0,0), Size(0,0), locations);

cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
cout << "Nr of locations specified : " << locations.size() << endl;

imshow("d", img);

      if (waitKey(25) >= 0)
  break;


    }
    return 0;
}
