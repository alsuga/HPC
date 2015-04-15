#include <bits/stdc++.h>
#include <cuda.h>

#define BLOCK_SIZE 1024

using namespace std;

__global__ sum(int *d_A, int *d_B, int *d_C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n*n)
    d_C[i] = d_A[i] + d_B[i];
}

__global__ sumR(int *d_A, int *d_B, int *d_C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j = 0; j < n; j++)
    d_C[i*n + j] = d_A[i*n + j] + d_B[i*n + j];
}

__global__ sumC(int *d_A, int *d_B, int *d_C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j = 0; j < n; j++)
    d_C[j*n + i] = d_A[j*n + i] + d_B[j*n + i];
}


int main(){
  int *h_A, *h_B, *h_C;
  int n = 5, size = sizeof(int) * n*n;
  h_A = (int *)malloc(size);
  h_B = (int *)malloc(size);
  h_C = (int *)malloc(size);

  for(int i = 0; i < n*n; i++) {
    h_A[i] = 1;
    h_B[i] = 2;
  }
  int *d_A, *d_B, *d_C;

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  sum<<< ceil( (n*n) / (double)BLOCK_SIZE), BLOCK_SIZE >>> (d_A, d_B, d_C, n);
  sumR<<< ceil( n / (double)BLOCK_SIZE), BLOCK_SIZE >>> (d_A, d_B, d_C, n);
  sumC<<< ceil( n / (double)BLOCK_SIZE), BLOCK_SIZE >>> (d_A, d_B, d_C, n);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  free(h_A);
  free(h_B);
  free(h_C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
