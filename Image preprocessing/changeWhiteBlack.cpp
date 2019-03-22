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
const Vec3b HSV_YELLOW_LOWER = Vec3b(0, 120, 130);
const Vec3b HSV_YELLOW_UPPER = Vec3b(40, 255, 255);

const Vec3b HLS_YELLOW_LOWER = Vec3b(20, 120, 80);
const Vec3b HLS_YELLOW_UPPER = Vec3b(45, 200, 255);

const Vec3b HSV_BLUE_LOWER = Vec3b(80, 200, 65);
const Vec3b HSV_BLUE_UPPER = Vec3b(120, 255, 130);

const Vec3b HSV_RED_LOWER = Vec3b(0, 100, 100);
const Vec3b HSV_RED_UPPER = Vec3b(10, 255, 255);
const Vec3b HSV_RED_LOWER1 = Vec3b(160, 100, 100);
const Vec3b HSV_RED_UPPER1 = Vec3b(179, 255, 255);

const int MAX_SIZE = 3;

int unSharpMask[MAX_SIZE][MAX_SIZE] = { {0, -1, 0},
																							{-1,  5, -1},
																							{0, -1, 0} };

void UnsharpMaskFilter(Mat & input, Mat & output, int mask[][MAX_SIZE]);



int main(int, char**)
{
	Mat img_input, img_result, img_gray, img_canny;
	Mat dilated;


    //Load the Images
  Mat image_obj = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/curve.png", CV_LOAD_IMAGE_GRAYSCALE );
  Mat image_scene = imread("/home/suki/바탕화면/Traffic Sign Recognition/image/IMG_1608.jpg");

	resize( image_obj, image_obj, Size(200, 200), 0, 0, CV_INTER_NN );
	resize( image_scene, image_scene, Size( 200, 200), 0, 0, CV_INTER_NN );

	Mat temp_image_scene = image_scene.clone();

  imshow("origin", image_scene);
    //Check whether images have been loaded
  if( !image_obj.data){
    cerr<< " --(!) Error reading image1 " << endl;
    return -1;
  }
  if( !image_scene.data){
    cerr<< " --(!) Error reading image2 " << endl;
    return -1;
  }


  	int height = image_obj.rows;
  	int width = image_obj.cols;
    int size = height * width;

    for(int i = 0; i < height; i++){
      for(int j = 0; j < width; j++){
        if(image_obj.at<uchar>(i, j) == 255){
          image_obj.at<uchar>(i, j) = 0;
        }
      }
    }



  imshow("origin", image_obj);


	waitKey(0);


	return 0;
}
