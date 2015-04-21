#include <cv.h>
#include <highgui.h>
#include <bits/stdc++.h>

#define RED 2
#define GREEN 1
#define BLUE 0

typedef unsigned char uchar;

using namespace cv;
using namespace std;

void grises(uchar *rgbImage, uchar *grayImage, int width, int height) {
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++){
      grayImage[i*height + j] = rgbImage[(i*height + j)*3 + RED] * 0.299 + rgbImage[(i*height + j)*3 + GREEN] * 0.587\
                             + rgbImage[(i*height + j)*3 + BLUE] * 0.114;
    }
  }
}

int main( ) {

  Mat image;
  uchar *dataimage, *grayimage;
  image = imread( "img1.jpg",1);

  dataimage = image.data;

  Mat gray_image;

  Size s = image.size();
  int width = s.width;
  int height = s.height;
  cout<<width<<" "<<height<<endl;
  int sizeGray = sizeof(uchar)*width*height;
  grayimage = (uchar *)malloc(sizeGray);

  grises(dataimage, grayimage, width, height);

  gray_image.create(height, width, CV_8UC1);
  gray_image.data = grayimage;

  imwrite("./Gray_Image.jpg",gray_image);
  //cvtColor( image, gray_image, CV_BGR2GRAY );

  //imwrite( "1093222278.png", gray_image );

   return 0;
}
