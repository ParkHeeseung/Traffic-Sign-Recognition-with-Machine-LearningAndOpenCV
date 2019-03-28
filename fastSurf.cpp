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
const Vec3b HSV_BLUE_UPPER = Vec3b(120, 255, 130);

const Vec3b HSV_RED_LOWER = Vec3b(0, 100, 100);
const Vec3b HSV_RED_UPPER = Vec3b(10, 255, 255);
const Vec3b HSV_RED_LOWER1 = Vec3b(160, 100, 100);
const Vec3b HSV_RED_UPPER1 = Vec3b(179, 255, 255);

const int MAX_SIZE = 3;
const int minHessian = 400;

int unSharpMask[MAX_SIZE][MAX_SIZE] = { {0, -1, 0},
																							{-1,  5, -1},
																							{0, -1, 0} };

void UnsharpMaskFilter(Mat & input, Mat & output, int mask[][MAX_SIZE]);

void Binarization(Mat & input, Mat & output);

int main(int, char**)
{

	Mat image_obj, image_scene;

	Ptr <SURF> detector = SURF::create( minHessian );
	Ptr <SURF> extractor = SURF::create();

	//Load the Images
	image_obj = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/TrafficSign.png", CV_LOAD_IMAGE_GRAYSCALE);



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


	vector<KeyPoint> keypoints_obj;
	detector->detect( image_obj, keypoints_obj );

	Mat descriptors_obj;
	extractor->compute( image_obj, keypoints_obj, descriptors_obj );




	while(1){

		clock_t begin, end;
		begin = clock();

		image_scene = imread("/home/suki/바탕화면/Traffic Sign Recognition/image/IMG_1581.JPG");

		// resize( image_obj, image_obj, Size(200, 200), 0, 0, CV_INTER_NN );
		resize( image_scene, image_scene, Size( 400, 400), 0, 0, CV_INTER_NN );

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


		//RGB to HSV
	  Mat hsvImg;
	  cvtColor(image_scene, hsvImg, CV_BGR2HSV);

		imshow("hsvImg", hsvImg);

		//Binarization
		Mat binaryImg;
		Binarization(hsvImg, binaryImg);
		// imshow("binary", binaryImg);



		//find contours
		vector<vector<Point> > contours;
		findContours(binaryImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		//contour를 근사화한다.
		vector<Point2f> approx;
		vector<vector<Point> > goodContours;

		int max = -1;
		int index = -1;

		cout << "find contour : " << contours.size() << endl;
		for (size_t i = 0; i < contours.size(); i++){
			approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

			if(fabs(contourArea(Mat(approx))) > max){
				max = fabs(contourArea(Mat(approx)));
				index = i;
			}

		}

		int xMax = -1;
		int xMin = 9999;
		int yMax = -1;
		int yMin = 9999;

		// cout << "ROI_Value" << contours.at(index) << endl;

		vector <Point> temp = contours.at(index);

		for(size_t i = 0; i < temp.size(); i++){
			// cout << "Element => " << temp.at(i).x << endl;

			int tempX = temp.at(i).x;
			int tempY = temp.at(i).y;

			if(xMax < tempX){
				xMax = tempX;
			}
				else if(xMin > tempX){
					xMin = tempX;
				}

				if(yMax < tempY){
					yMax = tempY;
				}
				else if(yMin > tempY){
					yMin = tempY;
				}

			}

			// cout << "xMax : " << xMax << " " << "xMin : " << xMin << endl;
			// cout << "yMax : " << yMax << " " << "yMin : " << yMin << endl;

			goodContours.push_back(contours.at(index));

			int xLen = xMax - xMin;
			int yLen = yMax - yMin;

			int roiSize = xLen > yLen ? xLen : yLen;

			// cout << "roiSize : " << roiSize << endl;

			/////////////////






		Mat dst = Mat::zeros(image_scene.rows, image_scene.cols, CV_8UC1);


		drawContours( dst, goodContours,  -1, cv::Scalar(255), CV_FILLED);

		Mat fillter_img;
		image_scene.copyTo(fillter_img, dst);

		image_scene = fillter_img.clone();

		cvtColor(image_scene, image_scene, CV_BGR2GRAY);


		imshow("init", image_scene);

	////////////////////insert/////////////////
		Mat idealROI;


		idealROI = image_scene(Rect(xMin, yMin, xLen, yLen));

		// erode(binaryImg, binaryImg, Mat());


		// idealROI = image_scene(Rect(xMin , yMin,\
		// 	xMin + roiSize + roiSize / 3 < xMax ? xMin + roiSize + roiSize / 3 : 500 - xMin, \
		// 	yMin + roiSize + roiSize / 3 < yMax ? yMin + roiSize + roiSize / 3 : 500 - yMin));

			// cout << "hi : " << image_scene.rows << endl;

		resize( idealROI, idealROI, Size( 200, 200), 0, 0, CV_INTER_NN );




		Mat sharpImg(200, 200, CV_8UC1);

		// UnsharpMaskFilter(idealROI, sharpImg, unSharpMask);

		// GaussianBlur(idealROI, sharpImg, Size(5, 5), 0);
		// GaussianBlur(image_obj, image_obj, Size(5, 5), 0);





		image_scene = idealROI.clone();


			  	// int height = image_scene.rows;
			  	// int width = image_scene.cols;
			    // int size = height * width;
					//
			    // for(int i = 0; i < height; i++){
			    //   for(int j = 0; j < width; j++){
			    //     if(image_scene.at<uchar>(i, j) == 255){
			    //       image_scene.at<uchar>(i, j) = 0;
			    //     }
			    //   }
			    // }



	  // resize( image_obj, image_obj, Size( image_scene.cols, image_scene.rows), 0, 0, CV_INTER_NN );

	  // image_scene = image_scene(Rect(0, 0, image_scene.cols, image_scene.rows/3));

	  // resize( image_scene, image_scene, Size( image_scene.cols, image_scene.rows), 0, 0, CV_INTER_NN );

		//-- Step 1: Detect the keypoints using SURF Detector

	  vector<KeyPoint> keypoints_scene;
	  detector->detect( image_scene, keypoints_scene );

	    //-- Step 2: Calculate descriptors (feature vectors)
	  Mat descriptors_scene;
	  extractor->compute( image_scene, keypoints_scene, descriptors_scene );

	    //-- Step 3: Matching descriptor vectors using FLANN matcher

	  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	  vector< vector<DMatch> > matches;
	  matcher->knnMatch( descriptors_obj, descriptors_scene, matches, 2 );

	  const float ratio_thresh = 0.7f;
	  vector< DMatch > good_matches;

	  for(size_t i = 0; i < matches.size(); i++){
	    if(matches[i][0].distance < ratio_thresh * matches[i][1].distance){
	      good_matches.push_back(matches[i][0]);
	    }
	  }

	  Mat img_matches;


	  drawMatches( image_obj, keypoints_obj, image_scene, keypoints_scene,
	                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	     //-- Step 4: Localize the object
	  // vector<Point2f> obj;
	  // vector<Point2f> scene;
		//
	  // if(good_matches.size() >= 3){

	    // for( int i = 0; i < good_matches.size(); i++ ){
	    //   //-- Step 5: Get the keypoints from the  matches
	    //   obj.push_back( keypoints_obj [good_matches[i].queryIdx ].pt );
	    //   scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	    // }

	    // //-- Step 6:FindHomography
	    // Mat H;
			//
	    // try { H = findHomography(obj, scene, CV_RANSAC); } catch (Exception e) {}


	    // //-- Step 7: Get the corners of the object which needs to be detected.
	    // vector<Point2f> obj_corners(4);
	    // obj_corners[0] = cvPoint(0,0);
	    // obj_corners[1] = cvPoint( image_obj.cols, 0 );
	    // obj_corners[2] = cvPoint( image_obj.cols, image_obj.rows );
	    // obj_corners[3] = cvPoint( 0, image_obj.rows );
			//
	    // //-- Step 8: Get the corners of the object form the scene(background image)
	    // std::vector<Point2f> scene_corners(4);

	    //-- Step 9:Get the perspectiveTransform

			// if(H.data){
			// 	perspectiveTransform( obj_corners, scene_corners, H);
			// }
			//
	    // //-- Step 10: Draw lines between the corners (the mapped object in the scene - image_2 )
	    // line( img_matches, scene_corners[0] + Point2f( image_obj.cols, 0), scene_corners[1] + Point2f( image_obj.cols, 0), Scalar(0, 255, 0), 4 );
	    // line( img_matches, scene_corners[1] + Point2f( image_obj.cols, 0), scene_corners[2] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
	    // line( img_matches, scene_corners[2] + Point2f( image_obj.cols, 0), scene_corners[3] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
	    // line( img_matches, scene_corners[3] + Point2f( image_obj.cols, 0), scene_corners[0] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
			//


	  // }

	  //-- Step 11: Mark and Show detected image from the background
	  imshow("DetectedImage", img_matches );
	  waitKey(0);


		// imshow("s", dst);


		end = clock();

		double duration = (double)(end - begin) / CLOCKS_PER_SEC;

		cout << "sec : " << duration << endl;

		// waitKey(0);


	}






	return 0;
}

void UnsharpMaskFilter(Mat & input, Mat & output, int mask[][MAX_SIZE]){

	int height = input.rows;
	int width = input.cols;
  int size = height * width;

	for(int i = MAX_SIZE / 2; i < height - MAX_SIZE / 2; i++){
    for(int j = MAX_SIZE / 2; j < width - MAX_SIZE / 2; j++){
      float sum = 0;
      for(int x = -1; x <= 1; x++){
        for(int y = -1; y <= 1; y ++){
          sum += input.at<uchar>(i + x, j + y) * mask[MAX_SIZE / 2 + x][MAX_SIZE / 2 + y];
        }
      }
      output.at<uchar>(i, j) = sum + 128;
    }

	}
}

void Binarization(Mat & input, Mat & output){

	Mat redBinaryImg, redBinaryImg1, yellowBinaryImg, blueBinaryImg;

	inRange(input, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER, yellowBinaryImg);

	inRange(input, HSV_BLUE_LOWER, HSV_BLUE_UPPER, blueBinaryImg);

	output = yellowBinaryImg | blueBinaryImg;

}
