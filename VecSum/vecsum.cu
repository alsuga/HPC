#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

#define BLOCK_SIZE 1024

int sz(int a, int b) {
  return (a + b -1) / b;
}

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
  __shared__ int tmp[BLOCK_SIZE];
  size_t t = threadIdx.x;
  size_t i = blockDim.x * blockIdx.x + t;
  if(i < n)
    tmp[t] = A[i];
  else
    tmp[t] = 0;
  __syncthreads();
  for(size_t st = blockDim.x/2; st > 0; st >>= 1){
    if(t < st)
      tmp[t] += tmp[t + st];
    __syncthreads();
  }
  __syncthreads();
  if (t == 0) out[blockIdx.x] = tmp[0];
}

int main(){
  int n; cin>>n;
  cudaError_t error = cudaSuccess;
  int size = n * sizeof(int);
  int *h_A = (int *)malloc(size);
  int *h_tmp = (int *)malloc(size);
  int *h_B = (int *)malloc(sizeof(int));
  //int *h_C = (int *)malloc(sz(size,BLOCK_SIZE));
  int *h_C = (int *)malloc(size);
  int *d_A, *d_B;
  init(h_A, n, 1);
  *h_B = 0;
  //double a, b;
  clock_t t = clock();

  //Secuencial ***************
  vecSum(h_A, h_B, n);
  t = clock() - t;
  //a = ((float)t)/CLOCKS_PER_SEC;
  cout<<"secuential: "<<h_B[0]<<endl;

  //paralelo*****************
  t = clock();
  cout<<"size: "<<size<<endl;
  error = cudaMalloc(&d_A, size);
  if(error != cudaSuccess){
    printf("Error reservando memoria para d_A");
    exit(0);
  }
  error = cudaMalloc(&d_B, sz(n, BLOCK_SIZE) * sizeof(int));
  if(error != cudaSuccess){
    printf("Error reservando memoria para d_B");
    exit(0);
  }
  error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if(error != cudaSuccess){
    printf("Error copiando h_A en d_A");
    exit(0);
  }
  while(n > 1) {
    vecSumP<<< sz(n, BLOCK_SIZE), BLOCK_SIZE>>> (d_A, d_B, n);
    error = cudaGetLastError();
    if ( cudaSuccess != error ){
      cout<<"Error en el kernel!"<<endl;
      cout<<cudaGetErrorString( error )<<endl;
      exit(0);
    }
    n = sz(n, BLOCK_SIZE);
    error = cudaMemcpy(d_A, d_B, n * sizeof(int), cudaMemcpyDeviceToDevice);
    if(error != cudaSuccess){
      printf("Error copiando d_A en d_B");
      cout<<cudaGetErrorString( error )<<endl;
      exit(0);
    }
    cudaDeviceSynchronize();
  }
  //Allocate memory for device
 // cudaMalloc(&d_A, size);
 // cudaMalloc(&d_B, sizeof(int));

  //Copy Data from host to device
  //cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  //Blocks and Grids

  //Launch Kernel
  //vecSumP<<< (n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>> (d_A, d_B,n);
  cudaDeviceSynchronize();

  //Copy from device, free device memory
  cudaMemcpy (h_C, d_B, sizeof(int), cudaMemcpyDeviceToHost);


  //matMultP(A,B,D,size);
  t = clock() - t;
  cout<<"parallel: "<<h_C[0]<<endl;
  //b = ((float)t)/CLOCKS_PER_SEC;
  //cout<<(a / b)<<endl;
  //printmat(C,n);
  //printmat(D,n);

  //if(compare(C,D,n)) cout<<"Work :)"<<endl;
  //else cout<<"No work :("<<endl;

  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  return 0;
}
