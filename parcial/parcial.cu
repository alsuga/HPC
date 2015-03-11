#include<cuda.h>
#include <bits/stdc++.h>

#define TILE_WIDTH 32

using namespace std;

//matrix initialization
void init(int *A,int n, int d){
  for(int i = 0; i < n*n; i++)
    A[i] = d;
}

//matrix comparation
bool compare(int *A, int *B, int rows, int cols){
  for(int i = 0; i < rows*cols; i++)
    if(A[i] != B[i])
      return false;
  return true;
}

//print matrix
void printmat(int *A, int rows, int cols){
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      cout<<A[i*rows+j]<<" ";
    }
    cout<<endl;
  }
  cout<<endl;
}

//matrix multiplication
void matMult(int *h_A, int *h_B, int *h_C, int common, int Arows, int Bcols){
  int sum;
  for(int i = 0; i < Arows; i++)
    for(int j = 0; j < Bcols; j++){
      sum = 0;
      for(int k = 0; k < common; k++)
        sum += h_A[common*i + k] * h_B[Bcols*k + j];
      h_C[Bcols*i + j] = sum;
       // h_C[n*i + j] += h_A[n*i + k] * h_B[n*k + j];
    }
}


//Parallel kernel
__global__ void matMultPP (int *A, int *B, int *C, int n){
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < n and j < n){
    int sum = 0;
    for(int k = 0; k < n; ++k)
      sum += A[n*i + k] * B[n*k + j];
    C[n*i + j] = sum;
  }
}

//Parallel kernel (tiling)
__global__ void matrixMulKernelTiled(int *d_M, int *d_N, int *d_P, int width){
  __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  for(int m = 0; m < width / TILE_WIDTH; ++m){
    Mds[ty][tx] = d_M[row*width + m*TILE_WIDTH + tx];
    Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty) * width + col];
    __syncthreads();
    for(int k = 0; k < TILE_WIDTH; ++k){
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  d_P[row*width+col] = Pvalue;
}

int main(){
  int n; cin>>n;
  cout<<n<<endl;
  int size = n*n*sizeof(int);
  int *A = (int *)malloc(size);
  int *B = (int *)malloc(size);
  int *C = (int *)malloc(size);
  int *D = (int *)malloc(size);
  int *d_A, *d_B, *d_C;
  init(A,n,1);
  init(B,n,2);
  init(C,n,0);
  init(D,n,0);
  double a, b;
  clock_t t = clock();

  //Secuencial
  matMult(A,B,C,n);
  t = clock() - t;
  a = ((float)t)/CLOCKS_PER_SEC;
  cout<<a<<endl;
  int block_size = 32;

  //paralelo
  t = clock();

  //Allocate memory for device
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);
  //Copy Data from host to device
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  //Blocks and Grids

  dim3 dimBlock(block_size,block_size);
  dim3 dimGrid(ceil(n/(float)block_size),ceil(n/(float)block_size));

  //Launch Kernel
  matMultPP<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, n);
  cudaDeviceSynchronize();
  //Copy from device, free device memory
  cudaMemcpy (D, d_C, size, cudaMemcpyDeviceToHost);


  //matMultP(A,B,D,size);
  t = clock() - t;
  b = ((float)t)/CLOCKS_PER_SEC;
  cout<<b<<endl;
  cout<<(a/b)<<endl;
  //printmat(C,n);
  //printmat(D,n);

  //if(compare(C,D,n)) cout<<"Work :)"<<endl;
  //else cout<<"No work :("<<endl;

  free(A);
  free(B);
  free(C);
  free(D);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
