---
layout: post
title:  Parallel programming - SIMD intrinsics
date: 2024-10-18 07:59:00-0400
description:  
tags:  parallel programming  
categories: parallel programming
featured: false
---



[cpu xeon 6230n](https://ark.intel.com/content/www/us/en/ark/products/192450/intel-xeon-gold-6230n-processor-27-5m-cache-2-30-ghz.html)
Instruction set extensions: Intel® SSE4.2, Intel® AVX, Intel® AVX2, Intel® AVX-512

[cpu i7-7700k](https://ark.intel.com/content/www/us/en/ark/products/97129/intel-core-i7-7700k-processor-8m-cache-up-to-4-50-ghz.html)
Instruction set extensions: Intel® SSE4.1, Intel® SSE4.2, Intel® AVX2

Both support hyper-threading technology which means each hardware core
has two processing threads per physical core.


## Prog2: Vectorizing  code using SIMD intrinsics

[Assignment link](https://github.com/stanford-cs149/asst1?tab=readme-ov-file#program-2-vectorizing-code-using-simd-intrinsics-20-points)

Solution idea:

Just translate the `clampedExpSerial` the code to use SIMD.
Refer to this `absVector` and `absSerial` to see how translation works.


To deal with situation that total number of loop iterations is 
not a multiple of SIMD vector width we can 
set `maskAll` at the beginning of the function so that 
only valid values of input vector is used for SIMD computation.
```cpp
    if(i + VECTOR_WIDTH > N) {
      maskAll = _cs149_init_ones(remain_count);
    } else {
      maskAll = _cs149_init_ones();
    }

```

code:
```cpp
void clampedExpVector(float* values, int* exponents, float* output, int N) {

  //
  // CS149 STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  
  __cs149_vec_float x;
  __cs149_vec_int   y_exponents;
  __cs149_vec_float result;
  __cs149_vec_int zeros_int = _cs149_vset_int(0);
  __cs149_vec_int ones_int = _cs149_vset_int(1);
  __cs149_vec_float ones_float = _cs149_vset_float(1.0f);
  __cs149_mask maskAll, maskExponentIsZero, maskExponentNotZero;  
  int remain_count = N % VECTOR_WIDTH;
  // int first_valid_count = remain_count > 0 ? VECTOR_WIDTH - remain_count : VECTOR_WIDTH;
  for(int i=0; i < N; i+= VECTOR_WIDTH) {
    if(i + VECTOR_WIDTH > N) {
      maskAll = _cs149_init_ones(remain_count);
    } else {
      maskAll = _cs149_init_ones();
    }
    // CS149Logger.addLog("initones", maskAll, VECTOR_WIDTH);

    maskExponentIsZero = _cs149_init_ones(0);

    _cs149_vload_float(x, values+i, maskAll);
    _cs149_vload_int(y_exponents, exponents+i, maskAll);

    // if y== 0
    _cs149_veq_int(maskExponentIsZero, y_exponents, zeros_int, maskAll);
    maskExponentNotZero = _cs149_mask_not(maskExponentIsZero);

    // x == 1  if y_exponents == 0
    //
    _cs149_vmove_float(result, ones_float, maskExponentIsZero);      

    // else 
    // result = x;
    _cs149_vmove_float(result, x, maskExponentNotZero);
    // count = y -1;
    _cs149_vsub_int(y_exponents, y_exponents, ones_int, maskExponentNotZero);
    __cs149_mask maskExpNotZeroCnt;
    _cs149_vgt_int(maskExpNotZeroCnt, y_exponents,  zeros_int, maskAll);
    // while (count > 0)
    // result *= x ;
    // count--;
    while(_cs149_cntbits(maskExpNotZeroCnt)) {
      _cs149_vmult_float(result, result, x, maskExpNotZeroCnt);

      _cs149_vsub_int(y_exponents, y_exponents, ones_int, maskExponentNotZero);
      _cs149_vgt_int(maskExpNotZeroCnt, y_exponents,  zeros_int, maskAll);
    }


    // if( result > 9.999999f) {
    // result = 9.999999f;
    // }
    __cs149_mask mask_gt_9;
    __cs149_vec_float nine_float = _cs149_vset_float(9.999999f);
    _cs149_vgt_float(mask_gt_9, result,  nine_float, maskAll);
    _cs149_vmove_float(result, nine_float, mask_gt_9);

    // output[i] = result;
    _cs149_vstore_float(output + i, result, maskAll);

    
  }
}
```



Run ./myexp -s 10000 and sweep the vector width from 2, 4, 8, to 16. Record the resulting vector utilization. You can do this by changing the #define VECTOR_WIDTH value in CS149intrin.h. Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

Answer: 
The vector utilization decrease as `VECTOR_WIDTH` increases.
The reason I think it's that not all values in vector is used for computation as 
`VECTOR_WIDTH` increases.

So it's not a very good idea to have very large `VECTOR_WIDTH`?

vector width 2:
```
(base) ➜  prog2_vecintrin git:(master) ✗ ./myexp -s 10000
CLAMPED EXPONENT (required)
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              2
Total Vector Instructions: 167727
Vector Utilization:        88.7%
Utilized Vector Lanes:     297685
Total Vector Lanes:        335454
************************ Result Verification *************************
Passed!!!
```

vector width 4:
```
(base) ➜  prog2_vecintrin git:(master) ✗ ./myexp -s 10000
CLAMPED EXPONENT (required)
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              4
Total Vector Instructions: 97075
Vector Utilization:        86.2%
Utilized Vector Lanes:     334817
Total Vector Lanes:        388300
************************ Result Verification *************************
Passed!!!
```

vector width 8:
```
(base) ➜  prog2_vecintrin git:(master) ✗ ./myexp -s 10000
CLAMPED EXPONENT (required)
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              8
Total Vector Instructions: 52877
Vector Utilization:        85.0%
Utilized Vector Lanes:     359535
Total Vector Lanes:        423016
************************ Result Verification *************************
Passed!!!
```

vector width 16:
```
(base) ➜  prog2_vecintrin git:(master) ✗ ./myexp -s 10000
CLAMPED EXPONENT (required)
Results matched with answer!
****************** Printing Vector Unit Statistics *******************
Vector Width:              16
Total Vector Instructions: 27592
Vector Utilization:        84.4%
Utilized Vector Lanes:     372781
Total Vector Lanes:        441472
************************ Result Verification *************************
Passed!!!
```

