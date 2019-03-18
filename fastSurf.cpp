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
const Vec3b HSV_YELLOW_UPPER = Vec3b(40, 140, 255);

const Vec3b HLS_YELLOW_LOWER = Vec3b(20, 120, 80);
const Vec3b HLS_YELLOW_UPPER = Vec3b(45, 200, 255);

const Vec3b HSV_BLUE_LOWER = Vec3b(80, 200, 100);
const Vec3b HSV_BLUE_UPPER = Vec3b(120, 255, 160);

const Vec3b HSV_RED_LOWER = Vec3b(0, 100, 100);
const Vec3b HSV_RED_UPPER = Vec3b(10, 255, 255);
const Vec3b HSV_RED_LOWER1 = Vec3b(160, 100, 100);
const Vec3b HSV_RED_UPPER1 = Vec3b(179, 255, 255);



int main(int, char**)
{
	Mat img_input, img_result, img_gray, img_canny;
	Mat dilated;


    //Load the Images
  Mat image_obj = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/curve.png", CV_LOAD_IMAGE_GRAYSCALE );
  Mat image_scene = imread("/home/suki/바탕화면/Traffic Sign Recognition/image/IMG_1608.JPG");

	resize( image_obj, image_obj, Size(150, 150), 0, 0, CV_INTER_NN );
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


  Mat hsvImg, binaryImg, binaryImg1;

  cvtColor(image_scene, hsvImg, CV_BGR2HSV);
  imshow("hsv", hsvImg);
  cout << "hi" << endl;

  inRange(hsvImg, HSV_RED_LOWER, HSV_RED_UPPER, binaryImg);
  inRange(hsvImg, HSV_RED_LOWER1, HSV_RED_UPPER1, binaryImg1);

  binaryImg = binaryImg | binaryImg1;

	inRange(hsvImg, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER, binaryImg1);

	binaryImg = binaryImg | binaryImg1;

	inRange(hsvImg, HSV_BLUE_LOWER, HSV_BLUE_UPPER, binaryImg1);


	binaryImg = binaryImg | binaryImg1;

	imshow("blue", binaryImg);

	// for(int i = 0; i < 10; i++){
	//
	// 	dilate(binaryImg, binaryImg, Mat());
	// }


	Mat element11(11, 11, CV_8U, Scalar(1));

 morphologyEx(binaryImg, binaryImg,MORPH_CLOSE,element11);


	vector<vector<Point> > contours;
	findContours(binaryImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	vector<vector<Point> > goodContours;

///////////////
		//contour를 근사화한다.
		vector<Point2f> approx;

		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

			if (fabs(contourArea(Mat(approx))) > 10)  //면적이 일정크기 이상이어야 한다.
			{


				int size = approx.size();

	      cout << approx << endl;

				if(size > 3){

					goodContours.push_back(contours.at(i));





				}

			}

		}

		/////////////////









	Mat dst = Mat::zeros(image_scene.rows, image_scene.cols, CV_8UC1);

	drawContours( dst, goodContours,  -1, cv::Scalar(255), CV_FILLED);

	Mat fillter_img;
	image_scene.copyTo(fillter_img, dst);

	image_scene = fillter_img.clone();

	cvtColor(image_scene, image_scene, CV_BGR2GRAY);


	imshow("init", image_scene);




  // resize( image_obj, image_obj, Size( image_scene.cols, image_scene.rows), 0, 0, CV_INTER_NN );

  // image_scene = image_scene(Rect(0, 0, image_scene.cols, image_scene.rows/3));

  // resize( image_scene, image_scene, Size( image_scene.cols, image_scene.rows), 0, 0, CV_INTER_NN );

    //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;
  Ptr<SURF> detector = SURF::create( minHessian );

  vector<KeyPoint> keypoints_obj,keypoints_scene;
  detector->detect( image_obj, keypoints_obj );
  detector->detect( image_scene, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
  Ptr <SURF> extractor = SURF::create();
  Mat descriptors_obj, descriptors_scene;
  extractor->compute( image_obj, keypoints_obj, descriptors_obj );
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
  vector<Point2f> obj;
  vector<Point2f> scene;

  if(good_matches.size() >= 3){

    for( int i = 0; i < good_matches.size(); i++ ){
      //-- Step 5: Get the keypoints from the  matches
      obj.push_back( keypoints_obj [good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    //-- Step 6:FindHomography
    Mat H;

    try { H = findHomography(obj, scene, CV_RANSAC); } catch (Exception e) {}


    //-- Step 7: Get the corners of the object which needs to be detected.
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( image_obj.cols, 0 );
    obj_corners[2] = cvPoint( image_obj.cols, image_obj.rows );
    obj_corners[3] = cvPoint( 0, image_obj.rows );

    //-- Step 8: Get the corners of the object form the scene(background image)
    std::vector<Point2f> scene_corners(4);

    //-- Step 9:Get the perspectiveTransform
    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Step 10: Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( image_obj.cols, 0), scene_corners[1] + Point2f( image_obj.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( image_obj.cols, 0), scene_corners[2] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( image_obj.cols, 0), scene_corners[3] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( image_obj.cols, 0), scene_corners[0] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );



  }

  //-- Step 11: Mark and Show detected image from the background
  imshow("DetectedImage", img_matches );
  waitKey(0);


	imshow("s", dst);


	waitKey(0);


	return 0;
}
