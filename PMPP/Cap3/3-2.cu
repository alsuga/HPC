#include <cuda.h>
#include <bits/stdc++.h>
using namespace std;
#define __ ios_base::sync_with_stdio(false);cin.tie(NULL);
#define endl '\n'
#define BLOCK_SIZE 32;

__global__
void mult(int *d_a, int *d_b, int *d_c, int m) {
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int acum = 0;
  for(int k = 0; k < m; k++) {
    acum += d_a[i][k] * d_b[k][j];
  }
  d_c[i][j] = acum;
}

__host__
void init(int *a, int m, int val) {
  for(int i = 0; i < m*m; i++)
    a[i] = val;
}

__host__
void print(int *a, int m) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++)
      cout<<a[i*m + j]<<" ";
    cout<<endl;
  }
}

int main(){__
  int m, sz;
  cin>>m;
  sz = sizeof(int)* m * m;
  int *a, *b, *c;
  a = (int *)malloc(sz);
  b = (int *)malloc(sz);
  c = (int *)malloc(sz);
  init(a, m, 1);
  init(b, m, 2);
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sz);
  cudaMalloc(&d_b, sz);
  cudaMalloc(&d_c, sz);
  cudaMemcpy(d_a, a, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sz, cudaMemcpyHostToDevice);

  mult<<< ceil(m / (double)BLOCK_SIZE), BLOCK_SIZE >>(d_a, d_b, d_c, m);

  cudaMemcpy(c, d_c, m, cudaMemcpyDeviceToHost);
  print(c,m);
  return 0;
}
