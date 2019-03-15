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
const Vec3b HSV_YELLOW_LOWER = Vec3b(20, 20, 130);
const Vec3b HSV_YELLOW_UPPER = Vec3b(40, 140, 255);

const Vec3b HLS_YELLOW_LOWER = Vec3b(20, 120, 80);
const Vec3b HLS_YELLOW_UPPER = Vec3b(45, 200, 255);

void polygonRoi(Mat& img, Mat& img_ROI, vector <Point2f> &approx);


void setLabel(Mat& image, string str, vector<Point> contour)
{
	int fontface = FONT_HERSHEY_SIMPLEX;
	double scale = 0.5;
	int thickness = 1;
	int baseline = 0;

	Size text = getTextSize(str, fontface, scale, thickness, &baseline);
	Rect r = boundingRect(contour);

	Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	rectangle(image, pt + Point(0, baseline), pt + Point(text.width, -text.height), CV_RGB(200, 200, 200), FILLED);
	putText(image, str, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}



int main(int, char**)
{
	Mat img_input, img_result, img_gray, img_canny;
	Mat dilated;

	//이미지파일을 로드하여 image에 저장
	img_input = imread("/home/suki/바탕화면/Traffic Sign Recognition/image/20190314_075520_271.jpg", IMREAD_COLOR);
	if (img_input.empty())
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}


	//그레이스케일 이미지로 변환
	cvtColor(img_input, img_gray, COLOR_BGR2GRAY);

	Canny(img_gray, img_canny, 150, 270);

	dilate(img_canny, dilated, Mat());



	imshow ("canny ", dilated);

	//contour를 찾는다.
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(dilated, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	//contour를 근사화한다.
	vector<Point2f> approx;
	img_result = img_gray.clone();

	Mat dst = Mat::zeros(img_input.rows, img_input.cols, CV_8UC1);



	 drawContours( dst, contours,  -1, cv::Scalar(255), CV_FILLED);





	 Mat fillter_img;
	img_gray.copyTo(fillter_img, dst);

	dst = fillter_img.clone();

	imshow("s", dst);

	imshow("input", img_input);
	imshow("result", img_result);


	waitKey(0);


	return 0;
}

void polygonRoi(Mat& img, Mat& img_ROI, vector <Point2f> &approx) {


	vector <Point> roiPoint;

  for(int i = 0; i < approx.size(); i++){
    roiPoint.push_back(approx.at(i));
    cout << i << endl;
  }

	Mat roi(img.rows, img.cols, CV_8U, Scalar(0));

	fillConvexPoly(roi, roiPoint, Scalar(255));

	Mat fillter_img;
	img.copyTo(fillter_img, roi);

	img_ROI = fillter_img.clone();

}
