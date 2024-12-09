---
layout: post
title: Autodiff implementation - kernel and memory management
date: 2024-12-04 07:59:00-0400
description:  
tags: ml framework
categories: ml framework
featured: false
---




This is fun to implement

Never aware that we can do management in python language like this.

I always think that memory management in only done in cpp.

Interesting, so interesting.

```
    def memory_plan(self, feed_shapes):
        """Allocates tvm.nd.array for every node except feed_dict nodes.

        Implementation note:
        Easy Option: Alloc a tvm.nd.array per node that persists across run()

        Use self.node_to_arr_map to store node->tvm.nd.array mapping
        to allow mapping to persist across multiple executor.run().

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""


```


## What is placeholder in tvm ? 


In TVM, a `placeholder` is used to define an input tensor for a computation. It essentially acts as a symbolic representation of the input data, specifying the shape and optionally the data type. This is useful for defining the structure of the computation graph before actual data is provided.

Here's a simple example:

```python
import tvm
from tvm import te

# Define the shape of the input tensor
shape = (3, 4)
dtype = "float32"

# Create a placeholder tensor
A = te.placeholder(shape, dtype=dtype, name="A")

# Print the placeholder tensor
print("Placeholder tensor A:", A)
```

In this example:
- `te.placeholder` creates a tensor `A` with the specified shape `(3, 4)` and data type `"float32"`.
- This tensor can then be used in further computations, such as defining operations that use `A` as input.

The `placeholder` is a fundamental concept in TVM's Tensor Expression (TE) language, allowing you to build complex computation graphs by defining how data flows through various operations¹².

¹: [TVM Documentation](https://tvm.apache.org/docs/reference/api/python/te.html)
²: [TVM Tutorial on Tensor Expressions](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html)

Source: Conversation with Copilot, 12/5/2024
(1) tvm.te — tvm 0.19.dev0 documentation - The Apache Software Foundation. https://tvm.apache.org/docs/reference/api/python/te.html.
(2) TVM 自底向上（三）：TE 的概念和编译原理 - 知乎. https://zhuanlan.zhihu.com/p/534313816.
(3) Working with Operators Using Tensor Expression — tvm 0.19.dev0 .... https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html.
(4) TVM学习（六）细读前端 - 知乎 - 知乎专栏. https://zhuanlan.zhihu.com/p/346514871.
(5) External Tensor Functions — tvm 0.19.dev0 documentation. https://tvm.apache.org/docs/how_to/work_with_schedules/extern_op.html.


## tvm create nd.array with specified shape
```python
import tvm
from tvm import te

# Specify the shape and data type
shape = (3, 4)
dtype = "float32"

# Create an empty TVM NDArray with the specified shape
tvm_array = tvm.nd.empty(shape, dtype)

# Print the TVM NDArray
print("TVM NDArray with specified shape:", tvm_array)

```


## Difference between tvm and ncnn
tvm is a open source deep learning compiler for cpus, gpus and specialized hardware.

ncnn is a neural network inference framework optimized for mobile and embedded devices.


We can assume input shape of mat_mul is 2d in this task.


## matmul kernel for unknown input dimension shape
This code is from gpt.
```python
import tvm
from tvm import te

def make_flexible_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host, func_name, dtype="float32"):
    # Determine the shapes of the input tensors
    if len(shapeA) == 2:
        batch = None
        n, k = shapeA
    else:
        batch, n, k = shapeA
    
    if len(shapeB) == 2:
        _, m = shapeB
    else:
        _, k, m = shapeB
    
    if transposeA:
        shapeA = (batch, k, n) if batch else (k, n)
    if transposeB:
        shapeB = (batch, m, k) if batch else (m, k)
    
    # Create placeholders for the input tensors
    A = te.placeholder(shapeA, name='A', dtype=dtype)
    B = te.placeholder(shapeB, name='B', dtype=dtype)
    
    # Define the reduction axis
    k_axis = te.reduce_axis((0, k), name='k')
    
    # Compute the matrix multiplication based on transpose flags and dimensionality
    if batch:
        if transposeA and transposeB:
            C = te.compute(
                (batch, n, m),
                lambda b, i, j: te.sum(A[b, k_axis, i] * B[b, j, k_axis], axis=k_axis),
                name='C'
            )
        elif transposeA:
            C = te.compute(
                (batch, n, m),
                lambda b, i, j: te.sum(A[b, k_axis, i] * B[b, k_axis, j], axis=k_axis),
                name='C'
            )
        elif transposeB:
            C = te.compute(
                (batch, n, m),
                lambda b, i, j: te.sum(A[b, i, k_axis] * B[b, j, k_axis], axis=k_axis),
                name='C'
            )
        else:
            C = te.compute(
                (batch, n, m),
                lambda b, i, j: te.sum(A[b, i, k_axis] * B[b, k_axis, j], axis=k_axis),
                name='C'
            )
    else:
        if transposeA and transposeB:
            C = te.compute(
                (n, m),
                lambda i, j: te.sum(A[k_axis, i] * B[j, k_axis], axis=k_axis),
                name='C'
            )
        elif transposeA:
            C = te.compute(
                (n, m),
                lambda i, j: te.sum(A[k_axis, i] * B[k_axis, j], axis=k_axis),
                name='C'
            )
        elif transposeB:
            C = te.compute(
                (n, m),
                lambda i, j: te.sum(A[i, k_axis] * B[j, k_axis], axis=k_axis),
                name='C'
            )
        else:
            C = te.compute(
                (n, m),
                lambda i, j: te.sum(A[i, k_axis] * B[k_axis, j], axis=k_axis),
                name='C'
            )
    
    # Create a schedule for the computation
    s = te.create_schedule(C.op)
    
    # Apply optimizations: split, reorder, vectorize, parallel
    if batch:
        b, i, j = s[C].op.axis
        k = s[C].op.reduce_axis[0]
        bo, bi = s[C].split(b, factor=16)
        io, ii = s[C].split(i, factor=16)
        jo, ji = s[C].split(j, factor=16)
        ko, ki = s[C].split(k, factor=16)
        s[C].reorder(bo, io, jo, ko, bi, ii, ji, ki)
        s[C].vectorize(ki)
        s[C].parallel(bo)
    else:
        i, j = s[C].op.axis
        k = s[C].op.reduce_axis[0]
        io, ii = s[C].split(i, factor=16)
        jo, ji = s[C].split(j, factor=16)
        ko, ki = s[C].split(k, factor=16)
        s[C].reorder(io, jo, ko, ii, ji, ki)
        s[C].vectorize(ki)
        s[C].parallel(io)
    
    # Lower the schedule to generate the IR code
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    
    # Build the function
    func = tvm.build(s, [A, B, C], tgt, tgt_host, name=func_name)
    
    return func

# Example usage
tgt = "llvm"
tgt_host = "llvm"
func_name = "flexible_matrix_mul"
shapeA = (32, 128, 64)  # Batch size of 32
shapeB = (32, 64, 128)  # Batch size of 32
transposeA = False
transposeB = False
make_flexible_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host, func_name)

```


## conv2d tvm kernel 
Code is generated by gpt
```python
import tvm
from tvm import te, topi

def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    # Create placeholders for the input tensor and filter
    X = te.placeholder((N, C, H, W), name='X', dtype=dtype)
    F = te.placeholder((M, C, R, S), name='F', dtype=dtype)

    # Define the reduction axes
    rc = te.reduce_axis((0, C), name='rc')
    rr = te.reduce_axis((0, R), name='rr')
    rs = te.reduce_axis((0, S), name='rs')

    # Compute the convolution
    Y = te.compute(
        (N, M, H - R + 1, W - S + 1),
        lambda n, m, h, w: te.sum(X[n, rc, h + rr, w + rs] * F[m, rc, rr, rs], axis=[rc, rr, rs]),
        name='Y'
    )

    # Create a schedule for the computation
    s = te.create_schedule(Y.op)

    # Apply optimizations: split, reorder, vectorize, parallel
    n, m, h, w = s[Y].op.axis
    rc, rr, rs = s[Y].op.reduce_axis
    ho, hi = s[Y].split(h, factor=16)
    wo, wi = s[Y].split(w, factor=16)
    s[Y].reorder(n, m, ho, wo, hi, wi, rc, rr, rs)
    s[Y].vectorize(wi)
    s[Y].parallel(ho)

    # Lower the schedule to generate the IR code
    print(tvm.lower(s, [X, F, Y], simple_mode=True))

    # Build the function
    func = tvm.build(s, [X, F, Y], tgt, tgt_host, name=func_name)

    return func

# Example usage
tgt = "llvm"
tgt_host = "llvm"
func_name = "conv2d"
shapeX = (1, 3, 32, 32)  # Example input shape (N, C, H, W)
shapeF = (16, 3, 3, 3)   # Example filter shape (M, C, R, S)
make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name)


```


## Install tvm by building from source
Can not install tvm through pip. Have to download source code and build it myself.
I don't know why.


Folow steps in this [issue](https://github.com/apache/tvm/issues/13507) to compile locally.

Need to disable gtest inroder to pass cmake 

[Offical install document](https://tvm.apache.org/docs/install/from_source.html#install-from-source)


Finally finish  installing tvm after building locally. 

Run this command to verify
```bash
(uwsyml) ➜  tvm git:(main) python -c "import tvm; print(tvm.__file__)"
/mnt/nvme1n1/zt/tvm/python/tvm/__init__.py
```

nosetests does not use python in conda. It uses that in /usr/bin which is not what I want. 

It reports error that it can not find numpy which I have already installed in conda environment 


## Update code to use latest function in tvm instead of old function in tvm


New code:
```python


def test_matrix_elementwise_mul():
    shape = (500, 200)
    x = np.random.uniform(0, 10, size=shape).astype(dtype)
    y = np.random.uniform(0, 10, size=shape).astype(dtype)
    z = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx)
    arr_y = tvm.nd.array(y, ctx)
    arr_z = tvm.nd.array(z, ctx)
    elemwise_mul = tvm_op.make_elemwise_mul(shape, tgt, tgt_host, "elem_add")
    elemwise_mul(arr_x, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x * y, z, rtol=1e-5)


    
def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = te.placeholder(shape, dtype=dtype, name='A')
    B = te.placeholder(shape, dtype=dtype, name='B')
    C = te.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


```

Old code:
```python

def test_matrix_elementwise_mul_by_const():
    shape = (2000, 3000)
    x = np.random.uniform(0, 10, size=shape).astype(dtype)
    const_val = np.random.uniform(0, 10)
    y = np.zeros(shape).astype(dtype)
    arr_x = tvm.nd.array(x, ctx=ctx)
    arr_y = tvm.nd.array(y, ctx=ctx)
    elemwise_mul_by_const = tvm_op.make_elemwise_mul_by_const(shape, const_val, tgt, tgt_host, "elem_mul_by_const")
    elemwise_mul_by_const(arr_x, arr_y)
    y = arr_y.asnumpy()
    np.testing.assert_allclose(x * const_val, y, rtol=1e-5)


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    C = tvm.compute(A.shape, lambda *i: A(*i) * const_k)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f
```
