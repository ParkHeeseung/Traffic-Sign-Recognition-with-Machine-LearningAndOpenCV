#include <iostream>
#include <fstream>
#include <ctime>
#include <queue>
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

int main(){



  //Load the Images
  Mat image_obj = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/stop.png", CV_LOAD_IMAGE_GRAYSCALE );
  Mat image_scene = imread("/home/suki/바탕화면/Traffic Sign Recognition/image/다운로드.jpeg", CV_LOAD_IMAGE_GRAYSCALE );

  //Check whether images have been loaded
  if( !image_obj.data){
    cerr<< " --(!) Error reading image1 " << endl;
    return -1;
  }
  if( !image_scene.data){
    cerr<< " --(!) Error reading image2 " << endl;
    return -1;
  }

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

  const float ratio_thresh = 0.8f;
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

  return 0;
}
