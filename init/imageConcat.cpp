#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main(){


  Mat narrowLaneImg, crossworkImg, curveImg, dynamicObstacleImg, parkingImg, staticObstacleImg, uTurnImg;
  Mat output;


  narrowLaneImg = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/Narrow lane.png", CV_LOAD_IMAGE_GRAYSCALE );
  crossworkImg = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/crosswalk.png", CV_LOAD_IMAGE_GRAYSCALE );
  curveImg = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/curve.png", CV_LOAD_IMAGE_GRAYSCALE );
  dynamicObstacleImg = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/dynamicObstacle.png", CV_LOAD_IMAGE_GRAYSCALE );
  parkingImg = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/parking.png", CV_LOAD_IMAGE_GRAYSCALE );
  staticObstacleImg = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/static.png", CV_LOAD_IMAGE_GRAYSCALE );
  uTurnImg = imread( "/home/suki/바탕화면/Traffic Sign Recognition/image/uturn.png", CV_LOAD_IMAGE_GRAYSCALE );

  resize( narrowLaneImg, narrowLaneImg, Size(200, 200), 0, 0, CV_INTER_NN );
  resize( crossworkImg, crossworkImg, Size(200, 200), 0, 0, CV_INTER_NN );
  resize( curveImg, curveImg, Size(200, 200), 0, 0, CV_INTER_NN );
  resize( dynamicObstacleImg, dynamicObstacleImg, Size(200, 200), 0, 0, CV_INTER_NN );
  resize( parkingImg, parkingImg, Size(200, 200), 0, 0, CV_INTER_NN );
  resize( staticObstacleImg, staticObstacleImg, Size(200, 200), 0, 0, CV_INTER_NN );
  resize( uTurnImg, uTurnImg, Size(200, 200), 0, 0, CV_INTER_NN );

  imshow("1", narrowLaneImg);
  imshow("2", crossworkImg);
  imshow("3", curveImg);
  imshow("4", dynamicObstacleImg);
  imshow("5", parkingImg);
  imshow("6", staticObstacleImg);
  imshow("7", uTurnImg);


	hconcat(narrowLaneImg, crossworkImg, output);
  hconcat(output, curveImg, output);
  hconcat(output, dynamicObstacleImg, output);
  hconcat(output, parkingImg, output);
  hconcat(output, staticObstacleImg, output);
  hconcat(output, uTurnImg, output);


	// namedWindow("같은 높이 이미지 결합", WINDOW_AUTOSIZE);
	imshow("Traffic_Sign", output);

	waitKey(0);

	return 0;
}
