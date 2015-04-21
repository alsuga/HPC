#include <cv.h>
#include <highgui.h>
#include <bits/stdc++.h>

#define RED 2
#define GREEN 1
#define BLUE 0
#define MASK_WIDTH 3

typedef unsigned char uchar;

using namespace cv;
using namespace std;

uchar sol(int i, int j) {
  i = (i < 0)? 0 : i;
  i = (i > 254)? 254 : i;
  j = (j < 0)? 0 : j;
  j = (j > 255)? 255 : j;

  int out = round(sqrt(i*i + j*j));
  return (out > 255)? 255 : out;
}

void grises(uchar *rgbImage, uchar *grayImage, int width, int height) {
  for(int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++){
      grayImage[i*width + j] = rgbImage[(i*width + j)*3 + RED] * 0.299 + rgbImage[(i*width+ j)*3 + GREEN] * 0.587\
                             + rgbImage[(i*width + j)*3 + BLUE] * 0.114;
    }
  }
}

void sobel(uchar *grayImage, int mask[], uchar* sobelImage, int width, int height) {
  int tmp, s_row, s_col, pv1, pv2;
  for(int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++) {
      tmp = 0;
      pv1 = pv2 = 0;
      s_row = i - (MASK_WIDTH/2);
      s_col = j - (MASK_WIDTH/2);
      for(int mask_i = 0; mask_i < MASK_WIDTH; mask_i++) {
        for(int mask_j = 0; mask_j < MASK_WIDTH; mask_j++) {
          if(s_row + mask_i >= 0 and s_row + mask_i < height and s_col + mask_j >= 0 and s_col + mask_j < width) {
            tmp =  (int)grayImage[(s_row+mask_i)*width +(s_col+mask_j)];
            pv1 += tmp * mask[mask_i * MASK_WIDTH + mask_j];
            pv2 += tmp * mask[mask_j * MASK_WIDTH + mask_i];
          }

          /*if(s_row + mask_i >= 0 and s_row + mask_i < height and s_col + mask_j >= 0 and s_col + mask_j < width) {
            tmp = (int)grayImage[((s_row + mask_i) * width) + (s_col + j)];
            cout<<(s_row + mask_i)<<" - "<<(s_col + j)<<" -> "<<tmp<<endl;
            pv1 += tmp * mask[mask_i * MASK_WIDTH + mask_j];
            //pv2 += tmp * mask[mask_j * MASK_WIDTH + mask_i];
          }*/
        }
      }
      //cout<<(int)sol(pv1)<<" ";
      sobelImage[i*width + j] = sol(pv1, pv2);
    }
        //cout<<endl;
  }
  cout<<height<<" f "<<width<<" - "<<height*width<<endl;
}

int main( ) {

  Mat image;
  uchar *dataimage, *grayimage, *sobelimage;
  image = imread( "img1.jpg",1);
  int Mask[] = {-1,0,1,-2,0,2,-1,0,1};
  dataimage = image.data;

  Mat gray_image, sobel_image;

  Size s = image.size();
  int width = s.width;
  int height = s.height;
  cout<<width<<" "<<height<<endl;
  int sizeGray = sizeof(uchar)*width*height;
  grayimage = (uchar *)malloc(sizeGray);
  sobelimage = (uchar *)malloc(sizeGray);

  grises(dataimage, grayimage, width, height);
  sobel(grayimage, Mask, sobelimage, width, height);

  gray_image.create(height, width, CV_8UC1);
  gray_image.data = grayimage;

  imwrite("./Gray_Image.jpg",gray_image);

  sobel_image.create(height, width, CV_8UC1);
  sobel_image.data = sobelimage;

  imwrite("./Sobel_Image.jpg", sobel_image);
   return 0;
}
