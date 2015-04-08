#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

const int BLOCK_SIZE = 64;

//vector initialization
void init(int *A,int n, int d){
  for(int i = 0; i < n; i++)
    A[i] = d;
}

//vector summatory
void vecSum(int *h_A, int *h_B, int n){
  for(int i = 0; i < n; i++)
      *h_B += h_A[i];
}


//Parallel kernel
__global__ void vecSumP (int *A, int *out, int n){
  __shared__ float tmp[BLOCK_SIZE];
  size_t t = threadIdx.x;
  size_t i = blockDim.x * blockIdx.x + t;
  if(i < n)
    tmp[t] = A[i];
  else
    tmp[t] = 0;
  __syncthreads();
  for(size_t st = blockDim.x; st > 1; st >>= 1){
    __syncthreads();
    if(t < st)
      tmp[t] += tmp[t + st];
  }
  __syncthreads();
  if (t == 0) out[blockIdx.x] = tmp[0];
}

int main(){
  int n; cin>>n;
  int size = n * sizeof(int);
  int *h_A = (int *)malloc(size);
  int *h_B = (int *)malloc((size + BLOCK_SIZE - 1)/BLOCK_SIZE);
  int *h_C = (int *)malloc(size);
  int *d_A, *d_B;
  init(h_A, n, 1);
  *h_B = 0;
  //double a, b;
  clock_t t = clock();

  //Secuencial
  vecSum(h_A, h_B, n);
  t = clock() - t;
  //a = ((float)t)/CLOCKS_PER_SEC;
  cout<<*h_B<<endl;

  //paralelo
  t = clock();

  //Allocate memory for device
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, sizeof(int));

  //Copy Data from host to device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  //Blocks and Grids

  //Launch Kernel
  vecSumP<<< (n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>> (d_A, n);
  cudaDeviceSynchronize();

  //Copy from device, free device memory
  cudaMemcpy (h_C, d_A, size, cudaMemcpyDeviceToHost);


  //matMultP(A,B,D,size);
  t = clock() - t;
  cout<<"parallel"<<endl;
  //b = ((float)t)/CLOCKS_PER_SEC;
  for(int i = 0; i < n; i++)
    cout<<h_C[i]<<endl;
  //cout<<(a / b)<<endl;
  //printmat(C,n);
  //printmat(D,n);

  //if(compare(C,D,n)) cout<<"Work :)"<<endl;
  //else cout<<"No work :("<<endl;

  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  //cudaFree(d_B);
  return 0;
}
