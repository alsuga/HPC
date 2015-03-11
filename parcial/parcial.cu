#include <cuda.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH BLOCK_SIZE
//int BLOCK_SIZE, TILE_WIDTH;

using namespace std;

//Declarations :
//matrix initialization
void init(float *A, int n, int d);

//matrix comparation
bool compare(float *A, float *B, int n);

//print matrix
void printmat(float *A, int rows, int cols);

//sequential matrix multiplication
void matMult(float *h_A, float *h_B, float *h_C, int common, int Arows, int Bcols);

//pre kernel matrix multiplication
void prematMultP(float *A, float *B, float *C, int common, int Arows, int Bcols);

//pre kernel matrix tiling multiplication
void prematMultPTiled(float *A, float *B, float *C, int common, int Arows, int Bcols);

//Parallel kernel
__global__ void matMultP (float *d_A, float *d_B, float *d_C, int common, int Arows, int Bcols);

//Parallel kernel (tiling)
__global__ void matMultPTiled(float *d_A, float *d_B, float *d_C, int common, int Arows, int Bcols);

//End declarations

int main() {
  for(int i = 0; i < 10; i++){
    cout<<i+1<<endl;
    //cin>>BLOCK_SIZE;
    //TILE_WIDTH = BLOCK_SIZE;
    int Arows,common,Bcols;
    cin >> Arows >> common >> Bcols;

    int sizeA = Arows * common * sizeof(float);
    int sizeB = common * Bcols * sizeof(float);
    int sizeR = Arows * Bcols * sizeof(float);
    float *A = (float *)malloc(sizeA);
    float *B = (float *)malloc(sizeB);
    float *C = (float *)malloc(sizeR);
    float *D = (float *)malloc(sizeR);
    float *E = (float *)malloc(sizeR);

    init(A, Arows * common, 1);
    init(B, common * Bcols, 2);
    init(C, Arows * Bcols, 0);
    init(D, Arows * Bcols, 0);
    init(E, Arows * Bcols, 0);

    double a, b, c;

    //Sequential
    clock_t t = clock();
    matMult(A, B, C, common, Arows, Bcols);
    t = clock() - t;
    a = ((float)t) / CLOCKS_PER_SEC;
    //cout << "Tiempo secuencial:" << endl;
    cout << a << endl;

    //Parallel
    t = clock();
    prematMultP(A, B, D, common, Arows, Bcols);
    t = clock() - t;
    b = ((float)t) / CLOCKS_PER_SEC;
    //cout << "Tiempo paralelo: " << endl;
    cout << b << endl;
    //cout << "Acelero con X " << endl;
    cout << (a / b) << endl;

    //Parallel (tiling)
    t = clock();
    prematMultPTiled(A, B, E, common, Arows, Bcols);
    t = clock() - t;
    c = ((float)t) / CLOCKS_PER_SEC;
    //cout << "Tiempo paralelo con tilings: " << endl;
    cout << c << endl;
    //cout << "Acelero con X " << endl;
    cout << (a / c) << endl;

    //print matrix
    //printmat(C,Arows,Bcols);
    //printmat(D,Arows,Bcols);
    //printmat(E,Arows,Bcols);

    //checking
    //if(compare(C, D, Arows * Bcols) and compare(D, E, Arows * Bcols))
      //cout << "Ok :)" << endl;
    //else
      //cout << "No ok :(" << endl;

    //Free
    free(A);
    free(B);
    free(C);
    free(D);
  }
  return 0;
}


//Functions

//matrix initialization
void init(float *A,int n, int d) {
  for(int i = 0; i < n; i++)
    A[i] = d;
}

//matrix comparation
bool compare(float *A, int *B, int n) {
  for(int i = 0; i < n; i++)
    if(abs(A[i] - B[i]) > 0.01)
      return false;

  return true;
}

//print matrix
void printmat(float *A, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      cout << A[i * rows + j] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

//matrix multiplication
void matMult(float *h_A, float *h_B, float *h_C, int common, int Arows, int Bcols) {
  float sum;
  for(int i = 0; i < Arows; i++)
    for(int j = 0; j < Bcols; j++) {
      sum = 0;
      for(int k = 0; k < common; k++)
        sum += h_A[common * i + k] * h_B[Bcols * k + j];

      h_C[Bcols * i + j] = sum;
    }
}

//pre kernel matrix multiplication
void prematMultP(float *A, float *B, float *C, int common, int Arows, int Bcols) {
  int sizeA = Arows * common * sizeof(float);
  int sizeB = common * Bcols * sizeof(float);
  int sizeR = Arows * Bcols * sizeof(float);
  float *d_A, *d_B, *d_C;

  //Allocate memory for device
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeR);

  //Copy Data from host to device
  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

  //Blocks and Grids
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(ceil(Bcols / (float)BLOCK_SIZE), ceil(Arows / (float)BLOCK_SIZE));

  //Launch Kernel
  matMultP<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, common, Arows, Bcols);
  cudaDeviceSynchronize();

  //Copy from device, free device memory
  cudaMemcpy (C, d_C, sizeR, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

//pre kernel matrix tiling multiplication
void prematMultPTiled(float *A, float *B, float *C, int common, int Arows, int Bcols) {
  int sizeA = Arows * common * sizeof(float);
  int sizeB = common * Bcols * sizeof(float);
  int sizeR = Arows * Bcols * sizeof(float);
  float *d_A, *d_B, *d_C;

  //Allocate memory for device
  cudaMalloc(&d_A, sizeA);
  cudaMalloc(&d_B, sizeB);
  cudaMalloc(&d_C, sizeR);

  //Copy Data from host to device
  cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

  //Blocks and Grids
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(ceil(Bcols / (float)BLOCK_SIZE), ceil(Arows / (float)BLOCK_SIZE));

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
__global__ void matMultP (float *d_A, float *d_B, float *d_C, int common, int Arows, int Bcols) {
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;

  if(i < Arows and j < Bcols) {
    float sum = 0;

    for(int k = 0; k < common; ++k)
      sum += d_A[common * i + k] * d_B[Bcols * k + j];

    d_C[Bcols * i + j] = sum;
  }
}

//Parallel kernel (tiling)
__global__ void matMultPTiled(float *d_A, float *d_B, float *d_C, int common, int Arows, int Bcols) {
  __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  for(int m = 0; m < (common + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

    if(m * TILE_WIDTH + tx < common and row < Arows)
      Mds[ty][tx] = d_A[row * common + m * TILE_WIDTH + tx];
    else
      Mds[ty][tx] = 0;

    if(m * TILE_WIDTH + ty < common and col < Bcols)
      Nds[ty][tx] = d_B[(m * TILE_WIDTH + ty) * Bcols + col];
    else
      Nds[ty][tx] = 0;

    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }

    __syncthreads();
  }

  if(row < Arows and col < Bcols)
    d_C[row * Bcols + col] = Pvalue;
}
