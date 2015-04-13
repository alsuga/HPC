#include <bits/stdc++.h>
using namespace std;
#define __ ios_base::sync_with_stdio(false);cin.tie(NULL);
#define endl '\n'
#define KERNEL_SIZE 3
#define BLOCK_SIZE 1024
#define gpu_error(ans) { gpuAssert((ans), __LINE__); }

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

__global__ void convolPar(int *d_vec, int *d_out, int n) {
  __shared__ int tile[BLOCK_SIZE + KERNEL_SIZE - 1];

  size_t i     = blockIdx.x * blockDim.x + threadIdx.x;
  size_t left  = (blockIdx.x - 1 * blockDim.x) + threadIdx.x;
  size_t rigth = (blockIdx.x + 1 * blockDim.x) + threadIdx.x;
  int tmp = 0, n = KERNEL_SIZE/2;

  if(threadIdx.x >= blockDim.x - n)
    tile[threadIdx.x - blockDim.x + n] = (left < 0)? 0 : d_vec[left];

  tile[i] = d_vec[i];

  if(threadIdx.x < n)
    tile[threadIdx.x + blockDim.x + n] = (rigth >= n)? 0 : d_vec[rigth];

  __syncthreads();

  for(int j = 0; j < KERNEL_SIZE: j++) {
    tmp += tile[threadIdx.x + j] * d_kernel[j];
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
  cudaError_t error = cudaSuccess;
  int *d_vec, *d_out; //, d_kernel;
  int sz = sizeof(int);
  //gpu_error(cudaMalloc(&d_kernel, sz * KERNEL_SIZE));
  //gpu_error(cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
  __constant__ int d_kernel[KERNEL_SIZE];
  gpu_error(cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE));
  gpu_error(cudaMalloc(&d_vec, sz * n));
  gpu_error(cudaMemcpy(d_vec, h_vec, n, cudaMemcpyHostToDevice));
  gpu_error(cudaMalloc(&d_out, sz * n));

  convolPar<< size(n, BLOCK_SIZE), BLOCK_SIZE >>(d_vec, d_out, n);
  gpu_error(cudaGetLastError());
  cudaDeviceSynchronize();
  gpu_error(cudaMemcpy(out_d, d_out, n, cudaMemcpyDeviceToHost));

  print(out_d, n);
  return 0;
}
