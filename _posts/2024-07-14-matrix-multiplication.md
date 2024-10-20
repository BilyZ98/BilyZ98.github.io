---
layout: post
title: Speed up matrix multiplication   
date: 2024-07-12 07:59:00-0400
description:  
tags:  al ml 
categories: ml
featured: false
---



https://gist.github.com/chris124567/c45d46fdf4d922389641cc9f591ae577

### Naive matrix multiplication 

Code 
```cpp
void matmul_byhand(float* input, float* weight, float* out, int N, int M, int K) {
	for(int input_row_idx=0; input_row_idx < N; input_row_idx++) {
		for(int output_col_idx =0 ; output_col_idx < K; output_col_idx++) {
      float sum = 0.0;
      for(int m_idx =0; m_idx < M; m_idx++) {
        int input_idx = input_row_idx * M + m_idx;
        int weight_idx = m_idx *  K  + output_col_idx;
        sum += input[input_idx] * weight[weight_idx];
      }

      int out_idx = input_row_idx * K + output_col_idx;
      out[out_idx] = sum;
    }
	}
}


int main() {
  int N = 3;
  int M = 3;
  int K = 4;
  float A[N * M];
  float B[M * K];
  float C[N * K];

  for (int i = 0; i < N * M; i++) {
    A[i] = i;
  }
  for (int i = 0; i < M * K; i++) {
    B[i] = i;
  }

  matmul(A, B, C, N, M, K);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      printf("%f ", C[i * K + j]);
    }
    printf("\n");
  }

  matmul_byhand(A, B, C, N, M, K);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      printf("%f ", C[i * K + j]);
    }
    printf("\n");
  }

  return 0;

}
```

Explanation:
The most outer loop iterates over the rows of the input matrix. 

The second outer loop iterates over the columns of the output matrix. 
The inner loop iterates over the columns of the input matrix and the rows of the weight matrix. 
The inner loop calculates the dot product of the input row and the weight column and stores the result in the output matrix.

Please note that how index of input, weight and output matrix are calculated.
![matmul3 drawio](https://github.com/user-attachments/assets/dc21b2cc-20f6-4720-8fbb-5adc7855d4f0)



Output:
```
20.000000 23.000000 26.000000 29.000000
56.000000 68.000000 80.000000 92.000000
92.000000 113.000000 134.000000 155.000000
```


Issue:
Failed to install cuda 12.1.0
```
nano-gpt) [nsccgz_qylin_1@ln101%tianhe2-K matmul]$ conda install nvidia/label/cuda-12.1.0::cuda-toolkit
Channels:
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
 - defaults
 - nvidia/label/cuda-12.1.0
 - conda-forge
 - nvidia
 - pytorch
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: failed

InvalidSpec: The package "nvidia/linux-64::cuda-compiler==12.5.1=0" is not available for the specified platform
```


```bash
conda install -y -c nvidia cuda=12.1.0 cuda-tools=12.1.0 cuda-toolkit=12.1.0 cuda-version=12.1 cuda-command-line-tools=12.1.0 cuda-compiler=12.1.0 cuda-runtime=12.1.0
```
