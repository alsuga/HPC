#include <cv.h>
#include <highgui.h>
#include <bits/stdc++.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;
using namespace std;

void grises(unsigned char* rgbImage, unsigned char* greyImage, int width, int height) {
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++){
      greyImage[i*width + j] = rgbImage[(i*width + j)*3 + RED] * 0.299 + rgbImage[(i*width + j)*3 + GREEN] * 0.587\
                             + rgbImage[(i*width + j)*3 + BLUE] * 0.114;
    }
  }
}

int main( ) {

  Mat image;
  uchar* dataimage;
  image = imread( "img1.jpg");

  dataimage = image.data;

  Mat gray_image;
  //cvtColor( image, gray_image, CV_BGR2GRAY );

  //imwrite( "1093222278.png", gray_image );

   return 0;
}
