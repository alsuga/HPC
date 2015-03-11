#include<cuda.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH BLOCK_SIZE

using namespace std;

//matrix initialization
void init(int *A,int n, int d);

//matrix comparation
bool compare(int *A, int *B, int n);

//print matrix
void printmat(int *A, int rows, int cols);

//matrix multiplication
void matMult(int *h_A, int *h_B, int *h_C, int common, int Arows, int Bcols);

void prematMultP(int *A, int *B, int* C, int common, int Arows, int Bcols);

void prematMultPTiled(int *A, int *B, int *C, int common, int Arows, int Bcols);

//Parallel kernel
__global__ void matMultP (int *d_A, int *d_B, int *d_C, int common, int Arows, int Bcols);

//Parallel kernel (tiling)
__global__ void matMultPTiled(int *d_A, int *d_B, int *d_C, int common, int Arows, int Bcols);

int main(){
  int Arows,common,Bcols; cin>>Arows>>common>>Bcols;
  //cout<<n<<endl;
  int sizeA = Arows*common*sizeof(int);
  int sizeB = common*Bcols*sizeof(int);
  int sizeR = Arows*Bcols*sizeof(int);
  int *A = (int *)malloc(sizeA);
  int *B = (int *)malloc(sizeB);
  int *C = (int *)malloc(sizeR);
  int *D = (int *)malloc(sizeR);
  int *E = (int *)malloc(sizeR);
  init(A,Arows*common,1);
  init(B,common*Bcols,2);
  init(C,Arows*Bcols,0);
  init(D,Arows*Bcols,0);
  init(E,Arows*Bcols,0);
  double a, b, c;

  //Secuencial
  clock_t t = clock();
  matMult(A,B,C,common,Arows,Bcols);
  t = clock() - t;
  a = ((float)t)/CLOCKS_PER_SEC;
  cout<<"Tiempo secuencial: "<<a<<endl;

  //paralelo
  t = clock();
  prematMultP(A,B,D,common,Arows,Bcols);
  t = clock() - t;
  b = ((float)t)/CLOCKS_PER_SEC;
  cout<<"Tiempo paralelo: "<<b<<endl;
  cout<<"Acelero "<<(a/b)<<" X"<<endl;
  t = clock();
  prematMultPTiled(A,B,E,common,Arows,Bcols);
  t = clock() - t;
  c = ((float)t)/CLOCKS_PER_SEC;
  cout<<"Tiempo paralelo con tilings: "<<c<<endl;
  cout<<"Acelero "<<(a/c)<<" X"<<endl;
  //printmat(C,Arows,Bcols);
  //printmat(D,Arows,Bcols);
  //printmat(E,Arows,Bcols);

  if(compare(C,D,Arows*Bcols) and compare(D,E,Arows*Bcols)) cout<<"Work :)"<<endl;
  else cout<<"No work :("<<endl;

  free(A);
  free(B);
  free(C);
  free(D);
  return 0;
}

//matrix initialization
void init(int *A,int n, int d){
  for(int i = 0; i < n; i++)
    A[i] = d;
}

//matrix comparation
bool compare(int *A, int *B, int n){
  for(int i = 0; i < n; i++)
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

void prematMultP(int *A, int *B, int* C, int common, int Arows, int Bcols){
  int sizeA = Arows*common*sizeof(int);
  int sizeB = common*Bcols*sizeof(int);
  int sizeR = Arows*Bcols*sizeof(int);
  int *d_A, *d_B, *d_C;
  //Allocate memory for device
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeR);
  //Copy Data from host to device
  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
  //Blocks and Grids

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(ceil(Bcols/(float)BLOCK_SIZE),ceil(Arows/(float)BLOCK_SIZE));

  //Launch Kernel
  matMultP<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, common, Arows, Bcols);
  cudaDeviceSynchronize();
  //Copy from device, free device memory
  cudaMemcpy (C, d_C, sizeR, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}


void prematMultPTiled(int *A, int *B, int *C, int common, int Arows, int Bcols){
  int sizeA = Arows*common*sizeof(int);
  int sizeB = common*Bcols*sizeof(int);
  int sizeR = Arows*Bcols*sizeof(int);
  int *d_A, *d_B, *d_C;
  //Allocate memory for device
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeR);
  //Copy Data from host to device
  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
  //Blocks and Grids

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(ceil(Bcols/(float)BLOCK_SIZE),ceil(Arows/(float)BLOCK_SIZE));

  //Launch Kernel
  matMultPTiled<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, common, Arows, Bcols);
  cudaDeviceSynchronize();
  //Copy from device, free device memory
  cudaMemcpy (C, d_C, sizeR, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

//Parallel kernel
__global__ void matMultP (int *d_A, int *d_B, int *d_C, int common, int Arows, int Bcols){
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  if(i < Arows and j < Bcols){
    int sum = 0;
    for(int k = 0; k < common; ++k)
      sum += d_A[common*i + k] * d_B[Bcols*k + j];
    d_C[Bcols*i + j] = sum;
  }
}

//Parallel kernel (tiling)
__global__ void matMultPTiled(int *d_A, int *d_B, int *d_C, int common, int Arows, int Bcols){
  __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  for(int m = 0; m < ( common + TILE_WIDTH - 1) / TILE_WIDTH; ++m){
    if(m*TILE_WIDTH + tx < common and row < Arows)
      Mds[ty][tx] = d_A[row*common + m*TILE_WIDTH + tx];
    else
      Mds[ty][tx] = 0;
    if(m*TILE_WIDTH + ty < common and col < Bcols)
      Nds[ty][tx] = d_B[(m*TILE_WIDTH + ty) * Bcols + col];
    else
      Nds[ty][tx] = 0;
    __syncthreads();
    for(int k = 0; k < TILE_WIDTH; ++k){
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  if(row < Arows and col < Bcols)
    d_C[row*Bcols+col] = Pvalue;
}


