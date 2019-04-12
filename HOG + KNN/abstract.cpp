#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main(){


  Mat frame = imread("hog_car_logos.jpg");

  for(int i = 0; i < 4; i ++){
    for(int j = 0; j < 4; j++){
      Mat rogo;
      rogo = frame(Rect(i * frame.rows / 4, j * frame.cols, frame.rows / 4, frame.cols / 4));
      imshow("data", rogo);
      waitKey(0);

    }
  }




	return 0;
}
