#include <cv.h>
#include <cuda.h>
#include <highgui.h>
#include <bits/stdc++.h>

#define RED 2
#define GREEN 1
#define BLUE 0
#define MASK_WIDTH 3
#define BLOCK_SIZE 32
#define TILE_WIDTH BLOCK_SIZE + MASK_WIDTH - 1
#define gpu_error(ans) { gpu_assert((ans), __LINE__); }

using namespace cv;
using namespace std;


__constant__ int d_maskc[MASK_WIDTH * MASK_WIDTH];

inline void gpu_assert(cudaError_t code, int line){
    if (code != cudaSuccess)
          cerr<<"GPUerror: "<<cudaGetErrorString(code)<<" in "<< line<<endl;
}

typedef unsigned char uchar;

__host__ __device__
uchar sol(int i, int j) {
  i = (i < 0)? 0 : i;
  i = (i > 254)? 254 : i;
  j = (j < 0)? 0 : j;
  j = (j > 255)? 255 : j;

  int out = sqrt((double)(i*i + j*j));
  return (out > 255)? 255 : out;
}

__global__
void D_grisesN(uchar *rgbImage, uchar *grayImage, int width, int height) {
  size_t i = blockIdx.y*blockDim.y+threadIdx.y;
  size_t j = blockIdx.x*blockDim.x+threadIdx.x;
  if((i < height) && (j < width))
  grayImage[i*width + j] = rgbImage[(i*width + j)*3 + RED] * 0.299 + rgbImage[(i*width+ j)*3 + GREEN] * 0.587\
                         + rgbImage[(i*width + j)*3 + BLUE] * 0.114;
}

__host__
void D_grises(uchar *h_rgbImage, uchar *h_grayImage, int width, int height) {
  uchar *d_rgbImage, *d_grayImage;
  int size = sizeof(uchar) * width * height;
  gpu_error(cudaMalloc(&d_rgbImage, size * 3 ));
  gpu_error(cudaMemcpy(d_rgbImage, h_rgbImage, size * 3, cudaMemcpyHostToDevice));
  gpu_error(cudaMalloc(&d_grayImage, size));
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 dimGrid(ceil(width/float(BLOCK_SIZE)),ceil(height/float(BLOCK_SIZE)),1);
  D_grisesN<<<dimGrid,dimBlock>>>(d_rgbImage, d_grayImage, width, height);
  cudaDeviceSynchronize();
  gpu_error(cudaMemcpy(h_grayImage, d_grayImage, size, cudaMemcpyDeviceToHost) );
  cudaFree(d_rgbImage);
  cudaFree(d_grayImage);
}



__host__
void H_grises(uchar *rgbImage, uchar *grayImage, int width, int height) {
  for(int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++){
      grayImage[i*width + j] = rgbImage[(i*width + j)*3 + RED] * 0.299 + rgbImage[(i*width+ j)*3 + GREEN] * 0.587\
                             + rgbImage[(i*width + j)*3 + BLUE] * 0.114;
    }
  }
}

__global__
void D_sobelN(uchar *grayImage, int *mask, uchar *sobelImage, int width, int height) {
  int tmp, s_row, s_col, pv1, pv2;
  size_t i = blockIdx.y*blockDim.y+threadIdx.y;
  size_t j = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < height and j < width) {
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
      }
    }
    sobelImage[i*width + j] = sol(pv1, pv2);
  }
}

__global__
void D_sobelC(uchar *grayImage, uchar *sobelImage, int width, int height) {
  int tmp, s_row, s_col, pv1, pv2;
  size_t i = blockIdx.y*blockDim.y+threadIdx.y;
  size_t j = blockIdx.x*blockDim.x+threadIdx.x;
  if(i < height and j < width) {
    tmp = 0;
    pv1 = pv2 = 0;
    s_row = i - (MASK_WIDTH/2);
    s_col = j - (MASK_WIDTH/2);
    for(int mask_i = 0; mask_i < MASK_WIDTH; mask_i++) {
      for(int mask_j = 0; mask_j < MASK_WIDTH; mask_j++) {
        if(s_row + mask_i >= 0 and s_row + mask_i < height and s_col + mask_j >= 0 and s_col + mask_j < width) {
          tmp =  (int)grayImage[(s_row+mask_i)*width +(s_col+mask_j)];
          pv1 += tmp * d_maskc[mask_i * MASK_WIDTH + mask_j];
          pv2 += tmp * d_maskc[mask_j * MASK_WIDTH + mask_i];
        }
      }
    }
    sobelImage[i*width + j] = sol(pv1, pv2);
  }
}

__global__
void D_sobelT(uchar *grayImage, int *mask, uchar *sobelImage, int width, int height) {
  int tmp, s_row, s_col, pv1, pv2;
  __shared__ int tile[TILE_WIDTH][TILE_WIDTH];
  int n = MASK_WIDTH/2;
  int row = blockIdx.y*blockDim.y+threadIdx.y - n;
  int col = blockIdx.x*blockDim.x+threadIdx.x - n;
  int trow = threadIdx.x;
  int tcol = threadIdx.y;
  //size_t ti = threadIdx.x *
  for(int i = 0; i < MASK_WIDTH; i++) {
    for(int j = 0; j < MASK_WIDTH; j++) {
      if(row + i < 0 or col + j < 0 or row + i >= height or col + i >= width)
        tile[i + trow][j + tcol] = 0;
      else
        tile[i + trow][j + tcol] = grayImage[(row + i)*width + (col + j)];
    }
  }
  if(row < height and col < width) {
    row += n;
    col += n;
    tmp = 0;
    pv1 = pv2 = 0;
    s_row = row - (MASK_WIDTH/2);
    s_col = col - (MASK_WIDTH/2);
    for(int mask_i = 0; mask_i < MASK_WIDTH; mask_i++) {
      for(int mask_j = 0; mask_j < MASK_WIDTH; mask_j++) {
        if(s_row + mask_i >= 0 and s_row + mask_i < height and s_col + mask_j >= 0 and s_col + mask_j < width) {
          tmp =  tile[s_row + mask_i][s_col + mask_j];
          pv1 += tmp * mask[mask_i * MASK_WIDTH + mask_j];
          pv2 += tmp * mask[mask_j * MASK_WIDTH + mask_i];
        }
      }
    }
    sobelImage[row*width + col] = sol(pv1, pv2);
  }
}

__host__
void D_sobel(uchar *grayImage, int mask[], uchar* sobelImage, int width, int height) {
  uchar *d_grayImage, *d_sobelImage;
  //int *d_mask; //global
  int size = sizeof(uchar) * width * height;
  gpu_error( cudaMalloc(&d_grayImage, size)  );
  gpu_error( cudaMemcpy(d_grayImage, grayImage, size, cudaMemcpyHostToDevice));
  gpu_error( cudaMalloc(&d_sobelImage, size) );
  gpu_error( cudaMemcpyToSymbol(d_maskc, mask, MASK_WIDTH * MASK_WIDTH * sizeof(int)) ); //cache

  //gpu_error( cudaMalloc(&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(int)) ); //global
  //gpu_error( cudaMemcpy(d_mask, mask, MASK_WIDTH * MASK_WIDTH * sizeof(int), cudaMemcpyHostToDevice)); //global

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 dimGrid(ceil(width/float(BLOCK_SIZE)),ceil(height/float(BLOCK_SIZE)),1);
  //D_sobelN<<<dimGrid,dimBlock>>>(d_grayImage, d_mask, d_sobelImage, width, height); //global
  D_sobelC<<<dimGrid,dimBlock>>>(d_grayImage, d_sobelImage, width, height);  //cache
  cudaDeviceSynchronize();
  gpu_error(cudaMemcpy(sobelImage, d_sobelImage, size, cudaMemcpyDeviceToHost) );
  cudaFree(d_grayImage);
  cudaFree(d_sobelImage);
  //cudaFree(d_mask); //global
}

__host__
void H_sobel(uchar *grayImage, int mask[], uchar* sobelImage, int width, int height) {
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
        }
      }
      sobelImage[i*width + j] = sol(pv1, pv2);
    }
  }
}


int main( ) {
  Mat image;
  double promSec = 0.0, promPar = 0.0;
  uchar *dataimage, *grayimage, *sobelimage;
  image = imread( "img1.jpg",1);
  int Mask[] = {-1, 0, 1, -2 , 0, 2, -1 ,0 ,1};
  dataimage = image.data;

  Mat gray_image, sobel_image;

  Size s = image.size();
  int width = s.width;
  int height = s.height;
  int sizeGray = sizeof(uchar)*width*height;

  grayimage = (uchar *)malloc(sizeGray);
  sobelimage = (uchar *)malloc(sizeGray);
  int n = 10;
  while(n--){
    clock_t t = clock();
    H_grises(dataimage, grayimage, width, height);
    H_sobel(grayimage, Mask, sobelimage, width, height);
    promSec += (clock() - t)/(float)CLOCKS_PER_SEC;

    t = clock();
    D_grises(dataimage, grayimage, width, height);
    D_sobel(grayimage, Mask, sobelimage, width, height);
    promPar += (clock() - t)/(float)CLOCKS_PER_SEC;
  }
  promSec /= 10;
  promPar /= 10;
  gray_image.create(height, width, CV_8UC1);
  gray_image.data = grayimage;

  imwrite("./Gray_Image.jpg",gray_image);

  sobel_image.create(height, width, CV_8UC1);
  sobel_image.data = sobelimage;

  imwrite("./Sobel_Image.jpg", sobel_image);
  cout<<"Secuencial:"<<endl;
  cout<<promSec<<endl;
  cout<<"Paralelo"<<endl;
  cout<<promPar<<endl;
  cout<<"Aceleracion"<<endl;
  cout<<promSec/promPar<<endl;
  return 0;
}
