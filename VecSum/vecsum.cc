#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

//vector initialization
void init(int *A,int n, int d){
  for(int i = 0; i < n; i++)
    A[i] = d;
}

//vector summatory
void vecSum(int *h_A, int *h_B, int n){
  for(int i = 0; i < n; i++)
      h_B += h_A[i];
}


//Parallel kernel
__global__ void vecSumP (int *A, int *B, int n){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n and i != 0)
    A[0] += A[i];
}

int main(){
  int n; cin>>n;
  int size = n * sizeof(int);
  int *h_A = (int *)malloc(size);
  int *h_B = (int *)malloc(sizeof int);
  int *h_C = (int *)malloc(sizeof int);
  int *d_A, *d_B;
  init(A, n, 1);
  &B = 0;
  double a, b;
  clock_t t = clock();

  //Secuencial
  vecSum(A, B, n);
  t = clock() - t;
  a = ((float)t)/CLOCKS_PER_SEC;
  cout<<a<<endl;
  int block_size = 32;

  //paralelo
  t = clock();

  //Allocate memory for device
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, sizeof int);

  //Copy Data from host to device
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  //Blocks and Grids

  //Launch Kernel
  matMultPP<<< (n + block_size - 1)/block_size, block_size>>> (d_A, d_B, n);
  cudaDeviceSynchronize();

  //Copy from device, free device memory
  cudaMemcpy (h_C, d_B, sizeof int, cudaMemcpyDeviceToHost);


  //matMultP(A,B,D,size);
  t = clock() - t;
  b = ((float)t)/CLOCKS_PER_SEC;
  cout<<b<<endl;
  cout<<(a / b)<<endl;
  //printmat(C,n);
  //printmat(D,n);

  //if(compare(C,D,n)) cout<<"Work :)"<<endl;
  //else cout<<"No work :("<<endl;

  free(A);
  free(B);
  cudaFree(d_A);
  cudaFree(d_B);
  return 0;
}
