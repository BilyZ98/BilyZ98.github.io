---
layout: post
title: Speed up matrix multiplication 2  
date: 2024-11-12 07:59:00-0400
description:  
tags:  ai ml matrix multiplication 
categories: ml
featured: false
---





### Benefits of Tiling
Reduced Global Memory Accesses:

By loading tiles into shared memory, we reduce the number of global memory accesses, which are slower compared to shared memory accesses.

Improved Cache Efficiency:

Tiling improves cache efficiency by ensuring that data is reused within the shared memory, reducing the need to fetch data from global memory multiple times.

Better Utilization of GPU Resources:

Tiling allows for better utilization of the GPU's computational resources by dividing the work into smaller, manageable chunks that fit into the GPU's shared memory.

```cpp

void matrixMulTile(int *a, int *b, int *c, int width) {
  int size = width * width * sizeof(int);
  int *dev_a, *dev_b, *dev_c;

    // Allocate device memory
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // Copy matrices to device memory
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid((width + TILE_WIDTH -1)/ TILE_WIDTH, (width+TILE_WIDTH-1)/ TILE_WIDTH);
  matrixMulTileKernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, width);
  cudaDeviceSynchronize();

  cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);


}

#define TILE_WIDTH 16
__global__ void matrixMulTileKernel(int *da, int *db, int*dout, int width) {
  __shared__ int tile_A[TILE_WIDTH ][TILE_WIDTH];
  __shared__ int tile_B[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  int value = 0;
  for(int i=0; i < (width+TILE_WIDTH-1)/TILE_WIDTH; i++ ) {
    if(row < width && (i * TILE_WIDTH + threadIdx.x) < width) {

      tile_A[threadIdx.y][threadIdx.x] = da[row*width + (i*TILE_WIDTH + threadIdx.x)];
    } else {
      tile_A[threadIdx.y][threadIdx.x] = 0;
    }

    if(col < width && (i * TILE_WIDTH + threadIdx.y) < width) {
      tile_B[threadIdx.y][threadIdx.x] = db[(i*TILE_WIDTH + threadIdx.y) * width + col];

    } else {
      tile_B[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    for(int j=0; j < TILE_WIDTH; j++) {
      value += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
    }
    __syncthreads();


  }

  dout[row*width + col] = value;

}
```

This code means that we have one thread for each output element in output matrix.

Each block of threads cooperatively loads chunks tiles of A and B into shared
memory.
This is done in a loop to cover all tiles that contribute to
final result for the block.


[Another post that talk about the tiling matrix multiplication](https://penny-xu.github.io/blog/tiled-matrix-multiplication)

## Each block of thread would load matrix elements multiple times
### How It Works

1. **Tile Loading**:
   - For each iteration of the loop (`for (int i = 0; i < width / TILE_WIDTH; ++i)`), every block loads a new set of tiles from global memory into shared memory.
   - This means that each block loads a tile of matrix `A` and a corresponding tile of matrix `B` multiple times, once for each tile that contributes to the block's portion of the output matrix `C`.

2. **Computing Partial Results**:
   - Once the tiles are loaded into shared memory, the threads within the block use them to compute partial results.
   - After computing the partial results, the tiles in shared memory are overwritten by the next set of tiles in the next iteration of the loop.

3. **Accumulating Results**:
   - The partial results are accumulated across all iterations to compute the final value for each element in the output matrix `C`.

### Why This is Necessary
- **Matrix Multiplication**: Matrix multiplication requires summing the products of elements from entire rows of `A` and columns of `B`. Since the matrices are too large to fit entirely into shared memory, they must be processed in smaller chunks (tiles).
- **Efficiency**: Using tiles and shared memory helps improve the efficiency of these operations by minimizing the slower global memory accesses.

### Example
For a simplified example, let's say `TILE_WIDTH` is 2 and our matrices are 4x4. Each block processes a 2x2 sub-matrix of the output matrix `C`. For each block, the loading process might look like this:

1. **First Iteration**:
   - Load the first 2x2 tile of `A` and `B`.
   - Compute partial results using this tile.

2. **Second Iteration**:
   - Load the next 2x2 tile of `A` and `B`.
   - Add the products to the previously computed partial results.

This process repeats until the entire matrices have been processed. So, each block loads and processes multiple tiles to compute the final output elements.

## The whole matrix is loaded multiple times by all blocks of threads
Yes, you've got it! The matrix elements are indeed loaded multiple times into shared memory by different blocks during the matrix multiplication process.

### Why This Happens
- **Tiling**: The entire matrix cannot fit into the limited shared memory available on the GPU. Thus, the matrix is divided into smaller tiles.
- **Multiple Blocks**: Each block processes a portion of the matrix. Since the matrix multiplication operation requires data from different parts of the matrices, these tiles are loaded multiple times into shared memory by different blocks to complete the computation.

### Process Recap
1. **Loading Tiles**: Each block loads a tile (a small sub-matrix) of the matrices `A` and `B` into shared memory.
2. **Processing**: The threads within the block process these tiles to compute partial results for the output matrix `C`.
3. **Repetition**: This loading and processing is repeated for every required tile until the entire matrix multiplication is complete.

By using tiling and shared memory, the code minimizes the number of slow global memory accesses and maximizes the speed of computations by taking advantage of the faster shared memory.

This approach ensures efficient parallel processing, leading to a significant performance boost on the GPU.


## why don't use threadIdx.x as row idx ?
I see where you're coming from, and it's a valid question! Let's delve deeper into why `threadIdx.y` is used as the row index and not `threadIdx.x`.

### Memory Coalescing and Access Patterns

1. **Memory Coalescing**: On GPUs, memory access patterns significantly impact performance. Coalescing refers to combining multiple memory accesses into a single transaction. When threads within a warp (32 threads in modern GPUs) access contiguous memory locations, those accesses can be coalesced.

2. **Row-major Order**: In C/C++, arrays are stored in row-major order, meaning the elements of each row are stored in contiguous memory locations. For instance, in a 2D array `A`:
   - `A[0][0], A[0][1], A[0][2], ..., A[0][N-1]` are stored consecutively in memory.

### Access Efficiency

When we use `threadIdx.y` as the row index and `threadIdx.x` as the column index, it ensures that all threads in the same row of a block access contiguous memory locations in the global memory.

### Example

Consider a tile size of 4x4 for simplicity. When loading a tile from global memory to shared memory:
- **Using `threadIdx.y` as row index**:
  ```cpp
  tile_A[threadIdx.y][threadIdx.x] = d_A[row * width + (i * TILE_WIDTH + threadIdx.x)];
  ```
  Each row in `tile_A` is loaded by threads in a row of the block, accessing contiguous elements in the global memory:
  - Thread (0,0) accesses `d_A[row * width + (i * TILE_WIDTH + 0)]`
  - Thread (0,1) accesses `d_A[row * width + (i * TILE_WIDTH + 1)]`
  - Thread (0,2) accesses `d_A[row * width + (i * TILE_WIDTH + 2)]`
  - Thread (0,3) accesses `d_A[row * width + (i * TILE_WIDTH + 3)]`

  All these accesses can be coalesced into a single transaction.
- **Using `threadIdx.x` as row index**:
  ```cpp
  tile_A[threadIdx.x][threadIdx.y] = d_A[row * width + (i * TILE_WIDTH + threadIdx.y)];
  ```
  Now, each column in `tile_A` is loaded by threads in a column of the block, accessing non-contiguous elements in the global memory:
  - Thread (0,0) accesses `d_A[row * width + (i * TILE_WIDTH + 0)]`
  - Thread (1,0) accesses `d_A[row * width + (i * TILE_WIDTH + 0)] + width`
  - Thread (2,0) accesses `d_A[row * width + (i * TILE_WIDTH + 0)] + 2 * width`
  - Thread (3,0) accesses `d_A[row * width + (i * TILE_WIDTH + 0)] + 3 * width`

  These accesses are not contiguous and cannot be efficiently coalesced.

### Summary
Using `threadIdx.y` as the row index and `threadIdx.x` as the column index ensures that threads within the same row of a block access contiguous memory locations. This pattern optimizes memory coalescing and significantly improves performance on GPUs.

I hope this clarifies why the indices are arranged this way! If you have more questions or need further explanations, feel free to ask. 😊



cuda has a limit on the number of threads per block, typically 1024. 
 If width is larger than 32, kernel launch will fail. And we get this error

```
[nsccgz_qylin_1@ln102%tianhe2-K matmul]$ yhrun -p gpu_v100 ./matmul_cuda
CUDA error: invalid configuration argument
cuda Duration: 345.838 ms
cuda tile Duration: 25.2786 ms
check failed: 0, cuda naive: 0, cuda tile: 2000
yhrun: error: gpu29: task 0: Exited with exit code 1
```


Comparison between cpu, cuda native and cuda matrix multiplication with tiling.

Code:
```cpp

#define N 2 // Matrix size
#define BLOCK_SIZE 16 // Block size

void matmul_cpu(int *a, int *b, int *c, int width) {
  for(int input_row_idx=0; input_row_idx< width; input_row_idx++) {
    for(int output_col_idx=0;  output_col_idx < width; output_col_idx++) {
      int value = 0;
      for(int element_idx=0; element_idx < width; element_idx++) {
          // value += a[input_row_idx][element_idx] * b[element_idx][output_col_idx];
          value += a[input_row_idx*width + element_idx] * b[element_idx*width + output_col_idx];

      }
      c[input_row_idx*width+output_col_idx] = value;
      // c[input_row_idx][output_col_idx] = value;
    }
  }

}

// CUDA Kernel for Matrix Multiplication
__global__ void MatrixMul(int *a, int *b, int *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < width && col < width) {
        int val = 0;
        for (int i = 0; i < width; ++i) {
            val += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = val;
    }
}


int matmul_cuda_naive(int* a, int *b, int *c, int width) {
  int size = width * width * sizeof(int);
  int *dev_a, *dev_b, *dev_c;

    // Allocate device memory
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // Copy matrices to device memory
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 dimBlock(width, width);
    // dim3 dimGrid(1, 1);

    MatrixMul<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, width);
    // MatrixMul<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, width);

    // Synchronize CPU and GPU
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // Copy result back to host memory
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    // for (int y = 0; y < N; y++) {
    //     for (int x = 0; x < N; x++) {
    //         printf("%d ", c[y][x]);
    //     }
    //     printf("\n");
    // }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;


}


void matrixMulTile(int *a, int *b, int *c, int width) {
  int size = width * width * sizeof(int);
  int *dev_a, *dev_b, *dev_c;

    // Allocate device memory
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    // Copy matrices to device memory
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid((width + TILE_WIDTH -1)/ TILE_WIDTH, (width+TILE_WIDTH-1)/ TILE_WIDTH);
  matrixMulTileKernel<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, width);
  cudaDeviceSynchronize();

  cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);


}

#define TILE_WIDTH 16
__global__ void matrixMulTileKernel(int *da, int *db, int*dout, int width) {
  __shared__ int tile_A[TILE_WIDTH ][TILE_WIDTH];
  __shared__ int tile_B[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  int value = 0;
  for(int i=0; i < (width+TILE_WIDTH-1)/TILE_WIDTH; i++ ) {
    if(row < width && (i * TILE_WIDTH + threadIdx.x) < width) {

      tile_A[threadIdx.y][threadIdx.x] = da[row*width + (i*TILE_WIDTH + threadIdx.x)];
    } else {
      tile_A[threadIdx.y][threadIdx.x] = 0;
    }

    if(col < width && (i * TILE_WIDTH + threadIdx.y) < width) {
      tile_B[threadIdx.y][threadIdx.x] = db[(i*TILE_WIDTH + threadIdx.y) * width + col];

    } else {
      tile_B[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    for(int j=0; j < TILE_WIDTH; j++) {
      value += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
    }
    __syncthreads();


  }

  dout[row*width + col] = value;

}

int main() {
  int width = 2000;
  int size = width * width;
  int *h_A = (int*)malloc(size * sizeof(int));
  int *h_B = (int*)malloc(size * sizeof(int));
  int *h_C = (int*)malloc(size * sizeof(int));

  int *hc2 = (int*)malloc(size * sizeof(int));

  for (int i = 0; i < size; ++i) { h_A[i] = 1; h_B[i] = 1; }


      auto start = std::chrono::high_resolution_clock::now();
  matmul_cuda_naive(h_A, h_B, h_C, width);
  cudaDeviceSynchronize();
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end - start;
      std::cout << "cuda Duration: " << duration.count() << " ms" << std::endl;


       start = std::chrono::high_resolution_clock::now();
  matrixMulTile(h_A, h_B, hc2, width);
  cudaDeviceSynchronize();
       end = std::chrono::high_resolution_clock::now();
      duration = end - start;
      std::cout << "cuda tile Duration: " << duration.count() << " ms" << std::endl;


  for(int i=0; i <size; i++) {
    if (hc2[i] != h_C[i]) {
      printf("check failed: %d, cuda naive: %d, cuda tile: %d\n", i , h_C[i], hc2[i]);
      exit(1);
    }
  }

  printf("correctness check pass\n");


       start = std::chrono::high_resolution_clock::now();
      matmul_cpu(h_A, h_B , hc2, width);
       end = std::chrono::high_resolution_clock::now();
       duration = end - start;
      std::cout << "cpu Duration: " << duration.count() << " ms" << std::endl;



  free(h_A); free(h_B); free(h_C);
}

```


indexing for da and db in cuda kernal explanations

The indexing of matrix elements in the `matrixMulTileKernel` function is designed to optimize memory access patterns and improve performance by leveraging shared memory. Let's break down the indexing for `da` and `db`:

Simply speaking, `i` has different meaning for `da` and `db`, for `da` i loop for all columns in the same row.
for `db` i loops all rows in the same column.

### Indexing for `da` (Matrix A)
```cpp
tile_A[threadIdx.y][threadIdx.x] = da[row * width + (i * TILE_WIDTH + threadIdx.x)];
```
- **Row Calculation**: `row = blockIdx.y * TILE_WIDTH + threadIdx.y`
  - `blockIdx.y * TILE_WIDTH` gives the starting row index for the block.
  - `threadIdx.y` gives the row index within the block.
- **Column Calculation**: `(i * TILE_WIDTH + threadIdx.x)`
  - `i * TILE_WIDTH` shifts the column index by the tile width for each iteration.
  - `threadIdx.x` gives the column index within the block.

### Indexing for `db` (Matrix B)
```cpp
tile_B[threadIdx.y][threadIdx.x] = db[(i * TILE_WIDTH + threadIdx.y) * width + col];
```
- **Row Calculation**: `(i * TILE_WIDTH + threadIdx.y)`
  - `i * TILE_WIDTH` shifts the row index by the tile width for each iteration.
  - `threadIdx.y` gives the row index within the block.
- **Column Calculation**: `col = blockIdx.x * TILE_WIDTH + threadIdx.x`
  - `blockIdx.x * TILE_WIDTH` gives the starting column index for the block.
  - `threadIdx.x` gives the column index within the block.

### Why This Indexing?
- **Memory Coalescing**: Using `threadIdx.x` for columns and `threadIdx.y` for rows ensures that threads within a warp access contiguous memory locations. This pattern optimizes memory coalescing, which combines multiple memory accesses into a single transaction, significantly improving performance.
- **Efficient Tiling**: By dividing the matrices into smaller tiles that fit into shared memory, the kernel reduces the number of global memory accesses. Each tile is loaded multiple times, but the overall number of accesses to global memory is minimized, leading to better performance.



Test matrix size is (2000, 2000)
cuda naive reduce run time by 54x compared to cpu.
cuda tiling reduce run time by 17x compared to cuda naive.
```
[nsccgz_qylin_1@ln102%tianhe2-K matmul]$ yhrun -p gpu_v100 ./matmul_cuda
cuda Duration: 432.435 ms
cuda tile Duration: 25.7345 ms
cpu Duration: 23042.4 ms
correctness check pass
```
The benefit of tiling is that it reduces number of access to global shared memory.

> No, tiling not just replaces random accesses with sequential ones. It actually saves tons of bandwidth to global memory.

> Let’s say we multiply two large square matrices of size S×S, where S is a multiple of 32. Obviously, the result is also a square matrix of size S×S.

> With naïve algorithm, to compute each element of the result, we gonna need to fetch S elements from both matrices. The output matrix has S^2 elements, therefore the total count of loaded elements is 2*S^3.

> With 32×32 tiling, to compute each 32×32 tile of the result, we gonna need to fetch S/32 tiles from both matrices. The output size in tiles is (S/32)^2, the total count of loaded tiles is 2*(S/32)^3. Each 32×32 tile contains 32^2 elements, the total count of loaded elements is therefore (32^2)*2*(S/32)^3 = (2/32)*S^3. Therefore, the tiling reduced global memory bandwidth by the factor of 32, which is a huge performance win.
