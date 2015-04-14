#include <bits/stdc++.h>
using namespace std;
#define __ ios_base::sync_with_stdio(false);cin.tie(NULL);
#define endl '\n'
#define KERNEL_SIZE 3
#define BLOCK_SIZE 4
#define gpu_error(ans) { gpu_assert((ans), __LINE__); }

__constant__ int d_cachekernel[KERNEL_SIZE];

inline void gpu_assert(cudaError_t code, int line){
  if (code != cudaSuccess)
    cerr<<"GPUerror: "<<cudaGetErrorString(code)<<" in "<< line<<endl;
}

int size(int n, int m){
  return (n + m -1) / m;
}

void print(int *vec, int n) {
  for(int i = 0; i < n; i++)
    cout << vec[i] << " ";
  cout<<endl;
}

void convolSec(int *vector, int *kernel, int *out, int n) {
  int tmp;
  for(int i = 0; i < n; i++) {
    out[i] = 0;
    tmp = i - KERNEL_SIZE/2;
    for(int j = 0; j < KERNEL_SIZE; j++) {
      if(tmp + j < 0 or tmp + j >= n)
        continue;
      out[i] += vector[tmp + j] * kernel[j];
    }
  }
}

__global__ void convolParB(int *d_vec, int *d_kernel, int *d_out, int n) {
  int i  = blockIdx.x * blockDim.x + threadIdx.x;

  int tmp = i - (KERNEL_SIZE/2), sum = 0;
  for(int j = 0; j < KERNEL_SIZE; j++) {
    if(tmp + j >= 0 and tmp + j < n)
      sum += d_vec[tmp + j] * d_kernel[j];
  }
  //__syncthreads();
  if(i < n)
    d_out[i] = sum;
}

__global__ void convolParB(int *d_vec, int *d_out, int n) {
  int i  = blockIdx.x * blockDim.x + threadIdx.x;

  int tmp = i - (KERNEL_SIZE/2), sum = 0;
  for(int j = 0; j < KERNEL_SIZE; j++) {
    if(tmp + j >= 0 and tmp + j < n)
      sum += d_vec[tmp + j] * d_cachekernel[j];
  }
  //__syncthreads();
  if(i < n)
    d_out[i] = sum;
}

__global__ void convolParT(int *d_vec, int *d_out, int n) {
  __shared__ int tile[BLOCK_SIZE + KERNEL_SIZE - 1];

  size_t i  = blockIdx.x * blockDim.x + threadIdx.x;
  int left  = ((blockIdx.x - 1) * blockDim.x) + threadIdx.x;
  int rigth = ((blockIdx.x + 1) * blockDim.x) + threadIdx.x;
  int wn = KERNEL_SIZE/2;

  if(threadIdx.x >= blockDim.x - wn)
    tile[threadIdx.x - (blockDim.x - wn)] = (left < 0)? 0 : d_vec[left];
    //tile[threadIdx.x - blockDim.x + wn] = (left < 0)? 0 : d_vec[left];

  tile[wn + threadIdx.x] = (i < n)? d_vec[i] : 0;

  if(threadIdx.x <= wn)
    tile[threadIdx.x + blockDim.x + wn] = (rigth >= n)? 0 : d_vec[rigth];

  __syncthreads();

  int tmp = 0;
  for(int j = 0; j < KERNEL_SIZE; j++) {
    tmp += tile[threadIdx.x + j] * d_cachekernel[j];
    //d_out[i]= tile[i+1];
  }
  d_out[i] = tmp;
}

int main(){__
  int *h_kernel = new int[KERNEL_SIZE];
  int *h_vec, *h_out, *out_d;
  int n; cin>>n;
  out_d = new int[n];
  h_vec = new int[n];
  h_out = new int[n];
  for(int i = 0; i < KERNEL_SIZE; i++)
    cin >> h_kernel[i];
  for(int i = 0; i < n; i++)
    cin >> h_vec[i];

  // <--------- Secuencial ----------->
  convolSec(h_vec, h_kernel, h_out, n);
  print(h_out, n);

  // <--------- Paralelo --------->
  int *d_vec, *d_out, *d_kernel;
  int sz = sizeof(int) * n;
  //gpu_error(cudaMalloc(&d_kernel, sizeof(int) * KERNEL_SIZE));
  //gpu_error(cudaMemcpy(d_kernel, h_kernel, sizeof(int)*KERNEL_SIZE, cudaMemcpyHostToDevice));
  gpu_error(cudaMemcpyToSymbol(d_cachekernel, h_kernel, sizeof(int) * KERNEL_SIZE));
  gpu_error(cudaMalloc(&d_vec, sz));
  gpu_error(cudaMemcpy(d_vec, h_vec, sz, cudaMemcpyHostToDevice));
  gpu_error(cudaMalloc(&d_out, sz ));

  //convolParB<<< size(n, BLOCK_SIZE), BLOCK_SIZE >>> (d_vec, d_kernel, d_out, n);
  convolParT<<< size(n, BLOCK_SIZE), BLOCK_SIZE >>> (d_vec, d_out, n);
  gpu_error(cudaGetLastError());
  cudaDeviceSynchronize();
  gpu_error(cudaMemcpy(out_d, d_out, sz, cudaMemcpyDeviceToHost));
  print(out_d, n);
  return 0;
}
