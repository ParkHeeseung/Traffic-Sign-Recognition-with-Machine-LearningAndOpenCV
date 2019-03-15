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

const Vec3b HSV_RED_LOWER = Vec3b(0, 100, 100);
const Vec3b HSV_RED_UPPER = Vec3b(10, 255, 255);
const Vec3b HSV_RED_LOWER1 = Vec3b(160, 100, 100);
const Vec3b HSV_RED_UPPER1 = Vec3b(179, 255, 255);



int main(int, char**)
{


  Mat img_input = imread("/home/suki/바탕화면/Traffic Sign Recognition/image/다운로드 (1).jpeg", IMREAD_COLOR);
	if (img_input.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

  imshow("origin", img_input);
  Mat hsvImg, binaryImg, binaryImg1;

  cvtColor(img_input, hsvImg, CV_BGR2HSV);
  imshow("hsv", hsvImg);

  inRange(hsvImg, HSV_RED_LOWER, HSV_RED_UPPER, binaryImg);
  inRange(hsvImg, HSV_RED_LOWER1, HSV_RED_UPPER1, binaryImg1);

  binaryImg = binaryImg | binaryImg1;

  dilate(binaryImg, binaryImg, Mat());


  vector<vector<Point> > contours;
  findContours(binaryImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  Mat dst = Mat::zeros(img_input.rows, img_input.cols, CV_8UC1);

  drawContours( dst, contours,  -1, cv::Scalar(255), CV_FILLED);


  Mat fillter_img;
  img_input.copyTo(fillter_img, dst);

  img_input = fillter_img.clone();

  imshow("input", img_input);




  imshow("binaryImg", binaryImg);
	waitKey(0);


	return 0;
}
