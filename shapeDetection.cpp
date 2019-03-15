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


int main(int, char**)
{
	Mat img_input, img_result, img_gray, img_canny;
	Mat dilated;

	//이미지파일을 로드하여 image에 저장
	img_input = imread("/home/suki/바탕화면/Traffic Sign Recognition/image/thumb_d_B34FFCCD9E4F41FA022E96D1BC5E9465.jpg", IMREAD_COLOR);
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

	vector<vector<Point> > contours;
	findContours(dilated, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat dst = Mat::zeros(img_input.rows, img_input.cols, CV_8UC1);

	drawContours( dst, contours,  -1, cv::Scalar(255), CV_FILLED);

	Mat fillter_img;
	img_gray.copyTo(fillter_img, dst);

	dst = fillter_img.clone();

	imshow("s", dst);
	imshow("input", img_input);


	waitKey(0);


	return 0;
}
