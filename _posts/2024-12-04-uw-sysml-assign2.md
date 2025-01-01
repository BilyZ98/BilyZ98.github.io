---
layout: post
title: Autodiff implementation - kernel and memory management
date: 2024-12-24 07:59:00-0400
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


### How node value is stored in autodiff ? 

It's stored in dict `node_to_val_map`

The graph is just computation graph. Each node is an operation node.

```python
def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array.

        Returns
        -------
        A list of values for nodes in eval_node_list. tvm.nd.array or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        node_to_val_map = {}
        for node, value in feed_dict.items():
            assert isinstance(value, tvm.ndarray.NDArray),\
                "feed_dict value type not supported"    
            node_to_val_map[node] = value

```


###  How node computation is done in graph ?

In run function , `compute` is called for each operation node.

Each opeartion node in computation graph has its own compute function.

For example , `AddOp` has its own compute function and this compute function
calls passed-in compiled_func to do the function call from compiled code.

Note that this `compiled_func` is built before forward of computation graph.

And the return function of `make_elemwise_add` is a tvm build function that 
takes `[A,B,C]` three tensor as input instead of parameters in `compute` function.


`tgt` and `shape` is defined during function compilation in `make_elemwise_add`

```python
    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array.

        Returns
        -------
        A list of values for nodes in eval_node_list. tvm.nd.array or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        node_to_val_map = {}
        for node, value in feed_dict.items():
            assert isinstance(value, tvm.ndarray.NDArray),\
                "feed_dict value type not supported"    
            node_to_val_map[node] = value

        # collect shapes for all placeholders
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            self.memory_plan(feed_shapes)
            self.compile_funcs(feed_shapes)

        # Traverse graph in topo order and compute values for all nodes.
        for node in self.topo_order:
            if node in node_to_val_map:
                # Skip placeholder nodes. Values already provided by feed_dict.
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            node_val = self.node_to_arr_map[node]
            # node_val is modified in-place
            node.op.compute(
                node, input_vals, node_val, self.node_to_compiled_func[node])
            node_to_val_map[node] = node_val
        # Collect node values.
        if convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
        return [node_to_val_map[n] for n in self.eval_node_list]


```
```python
class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, compiled_func):
        assert len(input_vals) == 2
        assert input_vals[0].shape == input_vals[1].shape
        compiled_func(input_vals[0], input_vals[1], output_val)  

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        return broadcast_rule(input_shapes[0], input_shapes[1])

    def compiled_func(self, node, input_shapes, tgt, tgt_host):
        return tvm_op.make_elemwise_add(
            input_shapes[0], tgt, tgt_host, "elem_add")


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = te.placeholder(shape, dtype=dtype, name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


```




### matrix_mul impl debug
Get this erro while debgging matrix_mul
```
======================================================================
ERROR: test_tvm_op.test_matrix_multiply
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/zt/miniconda3/envs/uwsyml/lib/python3.8/site-packages/nose/case.py", line 197, in runTest
    self.test(*self.arg)
  File "/mnt/nvme1n1/zt/assignment2-2018/tests/test_tvm_op.py", line 94, in test_matrix_multiply
    matrix_mul(arr_x, arr_y, arr_z)
  File "/mnt/nvme1n1/zt/tvm/python/tvm/runtime/module.py", line 201, in __call__
    return self.entry_func(*args)
  File "/mnt/nvme1n1/zt/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 245, in __call__
    raise_last_ffi_error()
  File "/mnt/nvme1n1/zt/tvm/python/tvm/_ffi/base.py", line 481, in raise_last_ffi_error
    raise py_err
tvm._ffi.base.TVMError: Traceback (most recent call last): 0: operator()                                                                                                                                                                                           
at /mnt/nvme1n1/zt/tvm/src/runtime/library_module.cc:82                                                                                                                                   
TVMError: Assert fail: T.Cast("int32", matrix_mul_B_shape[0]) == 500, Argument matrix_mul.B.shape[0] has an unsatisfied constraint: 500 == T.Cast("int32", matrix_mul_B_shape[0])   
```

Wrong code.
```python
def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    if transposeA:
        shapeA = (shapeA[1], shapeA[0])
    if transposeB:
        shapeB = (shapeB[1], shapeB[0])

    // This is wrong, we should put this code before if transposeA  
    A = te.placeholder(shapeA, dtype=dtype, name='A')
    B = te.placeholder(shapeB, dtype=dtype, name='B')

    assert shapeA[1] == shapeB[0]
    print("shape a 1",shapeA[1], "shapeB 0", shapeB[0])

    k = te.reduce_axis((0, shapeA[1]), name='k')
    if transposeA and transposeB:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B(j, k), axis=k),
                name='C'
                )
    elif transposeA  and (transposeB is False):
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B[k, j], axis=k),
                name='C'
                )
    elif (transposeA is False) and transposeB :
        print("come here")
        print('a shape ', A.shape, 'b shape', B.shape)
        assert(A.shape[1] == B.shape[1])
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[j, k], axis=k),
                name='C'
                )
    else:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                name='C'
                )


    s = te.create_schedule(C.op)

    # here to speed up matrix multiplication
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f



```



Correct code:
```python
def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    A = te.placeholder(shapeA, dtype=dtype, name='A')
    B = te.placeholder(shapeB, dtype=dtype, name='B')

    if transposeA:
        shapeA = (shapeA[1], shapeA[0])
    if transposeB:
        shapeB = (shapeB[1], shapeB[0])

    assert shapeA[1] == shapeB[0]
    print("shape a 1",shapeA[1], "shapeB 0", shapeB[0])

    k = te.reduce_axis((0, shapeA[1]), name='k')
    if transposeA and transposeB:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B(j, k), axis=k),
                name='C'
                )
    elif transposeA  and (transposeB is False):
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B[k, j], axis=k),
                name='C'
                )
    elif (transposeA is False) and transposeB :
        # print('a shape ', A.shape, 'b shape', B.shape)
        assert(A.shape[1] == B.shape[1])
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[j, k], axis=k),
                name='C'
                )
    else:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                name='C'
                )


    s = te.create_schedule(C.op)

    # here to speed up matrix multiplication
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

```

I asked gpt to find the bug in this code it does not find the bug.


### softmax cross entropy impl
Wrong code
```
def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""

    A = te.placeholder(shape, dtype=dtype, name='A')
    A_ = te.placeholder(shape, dtype=dtype, name='A_')

    B = te.compute(
           shape,
            lambda i, j: A_[i,j ] * te.log(A[i, j]),
            name='B'
            )

    row, col = shape
    axis_j = te.reduce_axis((0, col))
    axis_k = te.reduce_axis((0,row))
    C = te.compute(
            (1,),
            lambda :  -te.sum(B[ axis_j, axis_k], axis=[axis_j, axis_k]),
            name='C'
            )

    D = te.compute(
            (1,),
            lambda:  C / (row*col),
            name = 'D'
            )

    s = te.create_schedule(D.op)
    f = tvm.build(s, [A, A_, D], tgt, target_host=tgt_host, name=func_name)
    return f
```

Got this error with fowllowing code

error:
```
Traceback (most recent call last):
  File "/home/zt/miniconda3/envs/uwsyml/lib/python3.8/site-packages/nose/case.py", line 197, in runTest
    self.test(*self.arg)
  File "/mnt/nvme1n1/zt/assignment2-2018/tests/test_tvm_op.py", line 238, in test_softmax_cross_entropy
    matrix_softmax_cross_entropy = tvm_op.make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, "softmax_cross_entropy")
  File "/mnt/nvme1n1/zt/assignment2-2018/python/dlsys/tvm_op.py", line 217, in make_matrix_softmax_cross_entropy
    B = te.compute(

    ICHECK(0 == level_) << "Reductions are only allowed at the top level of compute. "
tvm.error.InternalError: Traceback (most recent call last):
  7: operator()
        at /mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc:168
  6: tvm::te::ComputeOp::ComputeOp(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::Map<tvm::runtime::String, tvm::runtime::ObjectRef, void, void>, tvm::runtime::Array<tvm::tir::IterVar, void>, tvm::runtime::Array<tvm::PrimExpr, void>)
        at /mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc:161
  5: VerifyComputeOp
        at /mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc:581
  4: Run
        at /mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc:551
  3: tvm::tir::ExprVisitor::VisitExpr_(tvm::tir::AddNode const*)
        at /mnt/nvme1n1/zt/tvm/src/tir/ir/expr_functor.cc:60
  2: VisitExpr
        at /mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc:560
  1: VisitExpr_
        at /mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc:564
  0: VisitExpr_
        at /mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc:566
  File "/mnt/nvme1n1/zt/tvm/src/te/operation/compute_op.cc", line 566
InternalError: Check failed: (0 == level_) is false: Reductions are only allowed at the top level of compute. Please create another tensor for further composition.
```
Code:

```python
def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""

    A = te.placeholder(shape, dtype=dtype, name='A')
    A_ = te.placeholder(shape, dtype=dtype, name='A_')
    row, col = shape
    axis_j = te.reduce_axis((0, col))
    axis_k = te.reduce_axis((0,row))

    B = te.compute(
           (row,),
           lambda i : -te.sum(A_[i, axis_j ] * te.log(A[i, axis_j]), axis=axis_j),
            name='B'
            )

    C = te.compute(
        (1,),
        lambda: te.sum(B[axis_k], axis=axis_k)/ row,
        name='C'
        )

```


I think the error is saying that we should not use reduction and te.log() together?

https://discuss.tvm.apache.org/t/non-top-level-reductions-in-compute-statements/5693

[softmax cross entropy reference impl](https://github.com/wyc-ruiker/CSE-599W-2018/blob/master/assignment2/python/dlsys/tvm_op.py)

Try another impl code

Fix code above after calculating te.log first and then do te.sum
```python
def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""

    A = te.placeholder(shape, dtype=dtype, name='A')
    A_ = te.placeholder(shape, dtype=dtype, name='A_')
    row, col = shape
    axis_j = te.reduce_axis((0, col), name='j')
    axis_k = te.reduce_axis((0,row), name='k')

    log = te.compute(
            shape,
            lambda i,j: te.log(A[i,j]),
            name='log'
            )
    sum_cross_entropy = te.compute(
            (row,),
            # lambda i: te.sum(B[i, axis_j], axis=axis_j ),
            lambda i: te.sum(-A[i, axis_j] * log[i, axis_j], axis=axis_j),
            name='sum_cross_entropy'
            )

    C = te.compute(
        (1,),
        lambda _: te.sum(sum_cross_entropy[axis_k]/row, axis=axis_k ) ,
        name='C'
        )
```


Correct code :
```python
def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""

    A = te.placeholder(shape, dtype=dtype, name='A')
    A_ = te.placeholder(shape, dtype=dtype, name='A_')

    row, col = shape
    softmax_axis_j = te.reduce_axis((0, col), name='softmax_j')
    softmax_axis_k = te.reduce_axis((0, col), name='softmax_k')
    max_x = te.compute(
            (shape[0], ), 
           lambda i: te.max(A[i,softmax_axis_j],axis=softmax_axis_j), 
           name= 'max_x')

    e_x = te.compute(
            shape,
            lambda i,j: te.exp(A[i,j ] - max_x[i]),
            name="e_x"
            )
    ex_sum = te.compute(
            (shape[0], ),
            lambda i: te.sum(e_x[i, softmax_axis_k], axis=softmax_axis_k),
            name='ex_sm')
    softmax = te.compute(
            shape,
            lambda i,j: e_x[i, j] / ex_sum[i],
            name='softmax_x')
 

    axis_j = te.reduce_axis((0, col), name='j')
    axis_k = te.reduce_axis((0,row), name='k')

    log = te.compute(
            shape,
            lambda i,j: te.log(softmax[i,j]),
            name='log'
            )
    sum_cross_entropy = te.compute(
            (row,),
            # lambda i: te.sum(B[i, axis_j], axis=axis_j ),
            lambda i: te.sum(-A_[i, axis_j] * log[i, axis_j], axis=axis_j),
            name='sum_cross_entropy'
            )

    C = te.compute(
        (1,),
        lambda _: te.sum(sum_cross_entropy[axis_k]/row, axis=axis_k ) ,
        name='C'
        )

    s = te.create_schedule(C.op)
    f = tvm.build(s, [A, A_, C], tgt, target_host=tgt_host, name=func_name)
    return f


```

### Fix infer_shape error 


`infer_shape()` is called for nodes in feed_dict. I don't know why this happens.

I am fixing it.
```
type of node <class 'dlsys.autodiff.Node'> node name X
Traceback (most recent call last):
  File "tests/mnist_dlsys.py", line 373, in <module>
    m(executor_ctx, num_epochs, print_loss_val_each_epoch)
  File "tests/mnist_dlsys.py", line 132, in mnist_logreg
    loss_val, grad_W1_val, grad_b1_val, _ = executor.run(
  File "/mnt/nvme1n1/zt/assignment2-2018/python/dlsys/autodiff.py", line 706, in run
    self.infer_shape(feed_shapes)
  File "/mnt/nvme1n1/zt/assignment2-2018/python/dlsys/autodiff.py", line 616, in infer_shape
    infer_shape = node.op.infer_shape(node, input_shapes)
  File "/mnt/nvme1n1/zt/assignment2-2018/python/dlsys/autodiff.py", line 320, in infer_shape
    assert False, "placeholder %s shape provided by feed_shape" % node.name
AssertionError: placeholder X shape provided by feed_shape
```

I fixed this issue by swapping `shape[0]` and `shape[1]` directly instead of calling `np.transpose`


### Speed up matrix multilication
Original matrix mul code and execution time
Code:
```python
def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    A = te.placeholder(shapeA, dtype=dtype, name='A')
    B = te.placeholder(shapeB, dtype=dtype, name='B')

    if transposeA:
        shapeA = (shapeA[1], shapeA[0])
    if transposeB:
        shapeB = (shapeB[1], shapeB[0])

    assert shapeA[1] == shapeB[0]
    print("shape a 1",shapeA[1], "shapeB 0", shapeB[0])

    k = te.reduce_axis((0, shapeA[1]), name='k')
    if transposeA and transposeB:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B(j, k), axis=k),
                name='C'
                )
    elif transposeA  and (transposeB is False):
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B[k, j], axis=k),
                name='C'
                )
    elif (transposeA is False) and transposeB :
        # print('a shape ', A.shape, 'b shape', B.shape)
        assert(A.shape[1] == B.shape[1])
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[j, k], axis=k),
                name='C'
                )
    else:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                name='C'
                )


    s = te.create_schedule(C.op)

    # here to speed up matrix multiplication
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


```
```
(uwsyml) ➜  assignment2-2018 git:(master) ✗ python tests/mnist_dlsys.py -l -m mlp
=== Build 3-layer MLP model...
Loading data...
Start training loop...
/mnt/nvme1n1/zt/tvm/python/tvm/driver/build_module.py:280: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  warnings.warn(
epoch 0
shape a 1 784 shapeB 0 784
shape a 1 256 shapeB 0 256
shape a 1 100 shapeB 0 100
shape a 1 10 shapeB 0 10
shape a 1 100 shapeB 0 100
shape a 1 1000 shapeB 0 1000
shape a 1 1000 shapeB 0 1000
shape a 1 1000 shapeB 0 1000
loss = 0.565684; Time taken this epoch = 39.259721 s
epoch 1
loss = 0.302340; Time taken this epoch = 37.834584 s
epoch 2
loss = 0.227699; Time taken this epoch = 37.836843 s
epoch 3
loss = 0.199743; Time taken this epoch = 37.733063 s
epoch 4
loss = 0.174254; Time taken this epoch = 37.731381 s
epoch 5
loss = 0.189644; Time taken this epoch = 37.791435 s
epoch 6
loss = 0.125607; Time taken this epoch = 37.795841 s
epoch 7
loss = 0.104398; Time taken this epoch = 37.821751 s
epoch 8
loss = 0.088052; Time taken this epoch = 37.845443 s
epoch 9
loss = 0.073229; Time taken this epoch = 37.798183 s
Validation set accuracy = 0.971600
Average Time per Training Epoch = 37.944825 s
```


Optmized code:
```python

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    A = te.placeholder(shapeA, dtype=dtype, name='A')
    B = te.placeholder(shapeB, dtype=dtype, name='B')

    if transposeA:
        shapeA = (shapeA[1], shapeA[0])
    if transposeB:
        shapeB = (shapeB[1], shapeB[0])

    assert shapeA[1] == shapeB[0]
    print("shape a 1",shapeA[1], "shapeB 0", shapeB[0])

    k = te.reduce_axis((0, shapeA[1]), name='k')
    if transposeA and transposeB:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B(j, k), axis=k),
                name='C'
                )
    elif transposeA  and (transposeB is False):
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[k, i] * B[k, j], axis=k),
                name='C'
                )
    elif (transposeA is False) and transposeB :
        # print('a shape ', A.shape, 'b shape', B.shape)
        assert(A.shape[1] == B.shape[1])
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[j, k], axis=k),
                name='C'
                )
    else:
        C = te.compute(
                (shapeA[0], shapeB[1]),
                lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                name='C'
                )


    s = te.create_schedule(C.op)
    x, y = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    xo, xi = s[C].split(x, factor=32)
    yo, yi = s[C].split(y, factor=32)
    ko, ki = s[C].split(k, factor=4)

    s[C].reorder(xo, yo, ko, xi, yi, ki)
    s[C].vectorize(yi)
    s[C].parallel(xo)


    # here to speed up matrix multiplication
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

```

Output:
```
(uwsyml) ➜  assignment2-2018 git:(master) ✗ python tests/mnist_dlsys.py -l -m mlp
=== Build 3-layer MLP model...
Loading data...
Start training loop...
/mnt/nvme1n1/zt/tvm/python/tvm/driver/build_module.py:280: UserWarning: target_host parameter is going to be deprecated. Please pass in tvm.target.Target(target, host=target_host) instead.
  warnings.warn(
epoch 0
shape a 1 784 shapeB 0 784
shape a 1 256 shapeB 0 256
shape a 1 100 shapeB 0 100
shape a 1 10 shapeB 0 10
shape a 1 100 shapeB 0 100
shape a 1 1000 shapeB 0 1000
shape a 1 1000 shapeB 0 1000
shape a 1 1000 shapeB 0 1000
loss = 0.565684; Time taken this epoch = 2.756940 s
epoch 1
loss = 0.302340; Time taken this epoch = 0.805593 s
epoch 2
loss = 0.227699; Time taken this epoch = 0.840440 s
epoch 3
loss = 0.199743; Time taken this epoch = 0.827800 s
epoch 4
loss = 0.174254; Time taken this epoch = 0.829301 s
epoch 5
loss = 0.189644; Time taken this epoch = 0.835602 s
epoch 6
loss = 0.125607; Time taken this epoch = 0.836001 s
epoch 7
loss = 0.104398; Time taken this epoch = 0.831005 s
epoch 8
loss = 0.088052; Time taken this epoch = 0.829629 s
epoch 9
loss = 0.073229; Time taken this epoch = 0.830049 s
Validation set accuracy = 0.971600
Average Time per Training Epoch = 1.022236 s
```

Explaination of `k`
Certainly! In the line `k = s[C].op.reduce_axis[0]`, we are accessing the first reduction axis of the operation that computes `C`.

Here's a breakdown:
- `s[C]`: This refers to the schedule for the computation of `C`.
- `s[C].op`: This accesses the operation that computes `C`.
- `s[C].op.reduce_axis`: This is a list of reduction axes used in the computation of `C`. Reduction axes are the dimensions over which we perform a reduction operation, such as summing elements.
- `[0]`: This accesses the first reduction axis in the list.

In the context of matrix multiplication, the reduction axis `k` represents the dimension over which we sum the products of elements from matrices `A` and `B`. For example, if `A` is of shape `(m, n)` and `B` is of shape `(n, p)`, the reduction axis `k` would range from `0` to `n-1`, summing the products of corresponding elements from `A` and `B` to compute each element of the resulting matrix `C`.

Let me know if you need any further clarification!



#### Problem Setup
Suppose `C` computes the result of multiplying two matrices \( A \) (size \( 64 \times 64 \)) and \( B \) (size \( 64 \times 64 \)):

\[
C[i, j] = \sum_{k=0}^{63} A[i, k] \cdot B[k, j]
\]

Here:
- `x` corresponds to `i` (rows of \( C \)),
- `y` corresponds to `j` (columns of \( C \)),
- `k` is the reduction axis (over \( k \)).

#### Steps in Context

1. **Splitting Axes**  
   - `x` (rows) and `y` (columns) are split into blocks of size `32`. This creates a **tile-based computation**:  
     - `xo` and `yo` iterate over \( 2 \times 2 \) blocks (as \( 64 / 32 = 2 \)).  
     - `xi` and `yi` handle elements within each \( 32 \times 32 \) block.
   - `k` is split into chunks of size `4`:
     - `ko` iterates over 16 chunks (as \( 64 / 4 = 16 \)).  
     - `ki` handles individual reduction operations within each chunk.

2. **Reordering**  
   The computation is reordered to maximize:
   - **Data locality**: Processing elements in nearby memory locations together.
   - **Parallelism**: Outer loops (`xo`, `yo`) can often run in parallel.

#### Execution Order
The reordered iteration could look like this (pseudocode):

```python
for xo in range(2):         # Iterate over 32-row blocks
    for yo in range(2):     # Iterate over 32-column blocks
        for ko in range(16): # Iterate over reduction chunks (k-axis)
            for xi in range(32):  # Process rows within a block
                for yi in range(32):  # Process columns within a block
                    for ki in range(4):  # Perform reduction within the chunk
                        C[xo*32 + xi, yo*32 + yi] += A[xo*32 + xi, ko*4 + ki] * B[ko*4 + ki, yo*32 + yi]
```

---

### Optimization Insight
This approach of splitting and reordering:
- **Improves memory access patterns**: Data is processed in small blocks, reducing cache misses.  
- **Enables parallel execution**: Larger outer loops (`xo`, `yo`, `ko`) can be distributed across threads or cores.
- **Reduces computation overhead**: By carefully controlling inner loops, computation can be streamlined for specific hardware.


This code snippet is an example of how to define a computation schedule in **TVM**, a machine learning compiler framework used for optimizing tensor computations. Let’s break it down step by step:

---

### 1. **`s = te.create_schedule(C.op)`**
- This creates a schedule for the operation (`C.op`) that needs to be optimized. 
- A schedule defines how the computation will be organized in terms of loops, parallelism, vectorization, etc.

---

### 2. **`x, y = s[C].op.axis`**
- `x` and `y` are the primary loop axes of the computation. These are typically the dimensions of the tensor being computed.
- For instance, if `C` is a 2D tensor, `x` and `y` might represent its row and column dimensions.

---

### 3. **`k = s[C].op.reduce_axis[0]`**
- `k` is a reduction axis, commonly used in operations like matrix multiplication or summation over an axis.
- If `C` is the result of a matrix multiplication \( A \times B \), then `k` would be the summation axis.

---

### 4. **`xo, xi = s[C].split(x, factor=32)`**
- Splits the `x` axis into two parts: 
  - `xo`: Outer loop, handling chunks of 32 iterations.
  - `xi`: Inner loop, handling the remaining iterations within each chunk.
- This is a technique for tiling the computation, which improves cache efficiency.

---

### 5. **`yo, yi = s[C].split(y, factor=32)`**
- Similar to `x`, this splits the `y` axis into outer (`yo`) and inner (`yi`) loops, with a tile size of 32.

---

### 6. **`ko, ki = s[C].split(k, factor=4)`**
- Splits the reduction axis `k` into:
  - `ko`: Outer loop for reduction.
  - `ki`: Inner loop for reduction, with a tile size of 4.

---

### 7. **`s[C].reorder(xo, yo, ko, xi, yi, ki)`**
- Rearranges the loop order to optimize computation.
- The new order ensures that the outer loops (`xo`, `yo`, `ko`) iterate first, followed by the inner loops (`xi`, `yi`, `ki`).
- This can improve locality of reference and allow efficient use of the hardware.

---

### 8. **`s[C].vectorize(yi)`**
- Applies vectorization to the `yi` loop. 
- This instructs the compiler to use SIMD (Single Instruction, Multiple Data) instructions to compute multiple iterations of the `yi` loop simultaneously, boosting performance.

---

### 9. **`s[C].parallel(xo)`**
- Parallelizes the `xo` loop, enabling multiple threads to compute different chunks of the `x` axis concurrently.
- This is particularly useful on multi-core processors.

---

### **Example**
Suppose `C` is the result of a matrix multiplication \( A[m, k] \times B[k, n] \), and we are optimizing the computation.

- **Matrix Dimensions**: \( A \) is \( 64 \times 128 \), \( B \) is \( 128 \times 64 \).
- The schedule:
  - Splits the computation into tiles of \( 32 \times 32 \) for `x` and `y` dimensions (row and column).
  - Reduces over `k` in chunks of 4.
  - Reorders loops to optimize memory access patterns.
  - Vectorizes operations within a 32-element column tile (`yi`).
  - Parallelizes over outermost rows (`xo`), enabling multi-threading.

This structured approach ensures better utilization of the CPU/GPU by leveraging parallelism, vectorization, and efficient memory access.



## We can parallelize xo and yo at the same time

Parallelizing `xo` distributes the workload of the outer `x` axis (rows of tiles) across multiple threads, allowing each thread to compute one or more row tiles simultaneously. This is often chosen because:

1. **Independent Workload**: Each tile along `xo` is independent, meaning computations for different tiles don’t depend on each other. This independence is a prerequisite for parallelization.
   
2. **Balanced Workload**: If the number of row tiles (determined by the size of `xo`) matches the number of available threads or cores, it results in a balanced workload.

3. **Memory Access**: For row-major memory layouts, accessing data row-wise tends to be more cache-friendly. Parallelizing `xo` aligns well with this layout.

---

### Can We Parallelize `yo` at the Same Time?

Yes, it's possible to parallelize `yo` as well, but there are important considerations:

1. **Nested Parallelism**: 
   - If `xo` and `yo` are both parallelized, this results in nested parallelism (threads inside threads).
   - This is often inefficient because hardware threads can't dynamically spawn or manage sub-threads efficiently, and most hardware supports a flat parallel structure.

2. **Overhead**: 
   - Parallelizing both `xo` and `yo` can introduce thread management overhead.
   - The system must divide and coordinate work across all threads, which can diminish performance if not carefully managed.

3. **Thread Count**:
   - Hardware has a limited number of threads or cores. If you parallelize both `xo` and `yo`, you risk oversubscribing threads, causing context-switching overhead.

---

### **When Parallelizing Both Could Work**
Parallelizing both `xo` and `yo` is feasible under specific conditions:

1. **Low Tile Counts in `xo` or `yo`**:
   - If `xo` alone doesn’t provide enough parallelism (e.g., when the matrix is small), you might also parallelize `yo`.

2. **Hierarchical Parallelism**:
   - You can map `xo` to thread blocks and `yo` to threads within each block on GPUs or similar architectures.

3. **Custom Scheduling**:
   - For some specialized cases, you can explicitly balance thread allocation across `xo` and `yo` to avoid over-parallelization.

---

### Example Scenario
If you're working with a 2D matrix computation and your target machine has 16 cores:
- **Parallelize only `xo`**: Efficient if the number of row tiles \( \frac{\text{rows}}{\text{tile\_size}} \) is ≥ 16.
- **Parallelize both `xo` and `yo`**: Useful when row and column tiles combined can provide enough parallelism (e.g., \( 4 \times 4 = 16 \) tiles).

To summarize, while it’s theoretically possible to parallelize both `xo` and `yo`, in practice, parallelizing only one is simpler and avoids excessive thread management overhead on most CPUs. For GPUs, however, hierarchical or hybrid parallelization across `xo` and `yo` is more common.
