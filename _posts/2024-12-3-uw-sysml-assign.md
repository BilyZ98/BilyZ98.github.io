---
layout: post
title: Autodiff implementation 
date: 2024-11-29 07:59:00-0400
description:  
tags: ml framework
categories: ml framework
featured: false
---


[Source assignment](https://github.com/dlsys-course/assignment1-2018)

Wrong code I wrote:
```

    for node in reverse_topo_order:
        cur_node_grad = node_to_output_grads_list[node]
        sum_grad = np.sum(cur_node_grad, axis=0)
        node_grad = node.op.gradient(node, sum_grad)
        for input in node.inputs:
            input_grads = node.op.gradient(input, sum_grad)
            if input not in node_to_output_grads_list:
                node_to_output_grads_list[input] = []
            node_to_output_grads_list[input].append(input_grads)


```

**Understanding the Code and Identifying the Issue**

The provided code implements a backpropagation algorithm to compute gradients of an output node with respect to a list of input nodes in a computational graph. The core idea is to traverse the graph in reverse topological order, calculating gradients at each node and propagating them backward.

However, the current implementation has a potential issue:

1. **Incorrect Gradient Calculation:**
   - The line `node_to_output_grad[node] = sum_grad` directly assigns the sum of gradients to the node's gradient. This is incorrect, as it doesn't account for the specific gradient contribution of each input to the node's output.
   - The subsequent gradient calculation for input nodes using `node.op.gradient` might also be affected by this incorrect sum.

**Corrected Code**

Here's a corrected implementation that addresses the issue:

```python
def gradients(output_node, node_list):
    """
    Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.
    """

    node_to_output_grads_list = {}
    node_to_output_grad = {}

    # Initialize gradient of output node as oneslike_op(output_node)
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]

    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        cur_node_grads = node_to_output_grads_list[node]
        sum_grad = np.sum(cur_node_grads, axis=0)  # Sum gradients across different output nodes

        # Calculate gradient for the current node
        node_grad = node.op.gradient(node, sum_grad)
        node_to_output_grad[node] = node_grad

        # Propagate gradients to input nodes
        for input_node, input_grad in zip(node.inputs, node_grad):
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            node_to_output_grads_list[input_node].append(input_grad)

    # Collect results for gradients requested
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list
```

**Explanation of the Correction**

1. **Correct Gradient Calculation:**
   - The `sum_grad` is now calculated by summing the gradients across different output nodes, ensuring that the total gradient for the current node is correctly computed.
   - The `node.op.gradient` method is used to calculate the gradient of the node with respect to its inputs, considering the current sum of gradients.
2. **Gradient Propagation:**
   - The gradients are propagated to input nodes using `zip` to iterate over input nodes and their corresponding gradients, ensuring correct association.

By addressing the incorrect gradient calculation, this revised code provides a more accurate and robust implementation of the backpropagation algorithm.


## nontype not iterable error 

Code:
```
def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    """TODO: Your code here"""
    for node in reverse_topo_order:
        cur_node_grad = node_to_output_grads_list[node]
        sum_grad = np.sum(cur_node_grad, axis=0)
        # gradient for each input node with gradient from current node
        node_grad = node.op.gradient(node, sum_grad)
        node_to_output_grad[node] = node_grad

        if node.inputs is not None:
            print('type node.inputs', type(node.inputs), 'node name', node.name,  'type node_grad', type(node_grad))
            for input_node, input_grad in zip(node.inputs, node_grad):
                if input_node not in node_to_output_grads_list:
                    node_to_output_grads_list[input_node] = []
                # This is wrong
                node_to_output_grads_list[input_node] = input_grad

        # node_grad = node.op.gradient(node, sum_grad)
        # for input in node.inputs:
        #     input_grads = node.op.gradient(input, sum_grad)
        #     if input not in node_to_output_grads_list:
        #         node_to_output_grads_list[input] = []
        #     node_to_output_grads_list[input].append(input_grads)

        # for i in range(len(node.inputs)):
        #     input_node = node.inputs[i]
        #     if input_node not in node_to_output_grads_list:
        #         node_to_output_grads_list[input_node] = []
        #     input_node_grad = node_grad[i]
        #     node_to_output_grads_list[input_node].append(input_node_grad)

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


```

Error:
```
======================================================================                                               ERROR: autodiff_test.test_add_mul_mix_1                                                                              ----------------------------------------------------------------------                                               Traceback (most recent call last):                                                                                     File "/home/zt/miniconda3/lib/python3.12/site-packages/nose/case.py", line 189, in runTest                             self.test(*self.arg)                                                                                               File "/mnt/nvme1n1/zt/assignment1-2018/autodiff_test.py", line 86, in test_add_mul_mix_1                               grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])                                                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                          File "/mnt/nvme1n1/zt/assignment1-2018/autodiff.py", line 353, in gradients                                            for input_node, input_grad in zip(node.inputs, node_grad):                                                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                        TypeError: 'NoneType' object is not iterable                                                                         -------------------- >> begin captured stdout << ---------------------
type node.inputs <class 'list'> node name (x1+((x2*x3)*x1)) type node_grad <class 'list'>
type node.inputs <class 'list'> node name ((x2*x3)*x1) type node_grad <class 'list'>
type node.inputs <class 'list'> node name (x2*x3) type node_grad <class 'list'>
type node.inputs <class 'list'> node name x3 type node_grad <class 'NoneType'>
                                                                                                                     --------------------- >> end captured stdout << ----------------------
-------------------- >> begin captured logging << --------------------
--------------------- >> end captured logging << ---------------------

```


I think it's because I don't add placeholder op to input x value.


Get another error 
```
ERROR: autodiff_test.test_add_by_const                                                                               ----------------------------------------------------------------------                                               Traceback (most recent call last):                                                                                     File "/home/zt/miniconda3/lib/python3.12/site-packages/nose/case.py", line 189, in runTest                             self.test(*self.arg)                                                                                               File "/mnt/nvme1n1/zt/assignment1-2018/autodiff_test.py", line 26, in test_add_by_const                                y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})                                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                        File "/mnt/nvme1n1/zt/assignment1-2018/autodiff.py", line 306, in run                                                  topo_order = find_topo_sort(self.eval_node_list)                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                   File "/mnt/nvme1n1/zt/assignment1-2018/autodiff.py", line 392, in find_topo_sort                                       topo_sort_dfs(node, visited, topo_order)                                                                           File "/mnt/nvme1n1/zt/assignment1-2018/autodiff.py", line 400, in topo_sort_dfs                                        for n in node.inputs:                                                                                                         ^^^^^^^^^^^                                                                                             AttributeError: 'NoneType' object has no attribute 'inputs'                                                          -------------------- >> begin captured stdout << ---------------------                                               type node.inputs <class 'list'> node name (x2+5) type node_grad <class 'list'>                                                                                                                                                            --------------------- >> end captured stdout << ----------------------                                               -------------------- >> begin captured logging << --------------------                                               --------------------- >> end captured logging << ---------------------
```


Correct code:
```python
def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    """TODO: Your code here"""
    for node in reverse_topo_order:
        if node in node_to_output_grads_list:
            node_to_output_grad[node] = sum_node_list(node_to_output_grads_list[node])
            grads = node.op.gradient(node, node_to_output_grad[node])
        # cur_node_grad = node_to_output_grads_list[node]
        # sum_grad = np.sum(cur_node_grad, axis=0)
        # gradient for each input node with gradient from current node
        # node_grad = node.op.gradient(node, sum_grad)
        # node_to_output_grad[node] = node_grad

        # if node_grad is not None:
        #     print('type node.inputs', type(node.inputs), 'node name', node.name,  'type node_grad', type(node_grad))
        #     for input_node, input_grad in zip(node.inputs, node_grad):
        #         if input_node not in node_to_output_grads_list:
        #             node_to_output_grads_list[input_node] = []
        #         node_to_output_grads_list[input_node] = input_grad
            for id , in_nodes in enumerate(node.inputs):
                if in_nodes not in node_to_output_grads_list:
                    node_to_output_grads_list[in_nodes] = []
                node_to_output_grads_list[in_nodes].append(grads[id])

        # node_grad = node.op.gradient(node, sum_grad)
        # for input in node.inputs:
        #     input_grads = node.op.gradient(input, sum_grad)
        #     if input not in node_to_output_grads_list:
        #         node_to_output_grads_list[input] = []
        #     node_to_output_grads_list[input].append(input_grads)

        # for i in range(len(node.inputs)):
        #     input_node = node.inputs[i]
        #     if input_node not in node_to_output_grads_list:
        #         node_to_output_grads_list[input_node] = []
        #     input_node_grad = node_grad[i]
        #     node_to_output_grads_list[input_node].append(input_node_grad)

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


```


I don't know why this is the correct code.

What is the difference between my previous code and this code ?


This is very different from pytorch like auto gradient calculation.

[Autodiff algorithm lecture](https://dlsys.cs.washington.edu/pdf/lecture4.pdf)

**Yes, PyTorch leverages automatic differentiation (autodiff) to calculate gradients automatically.** This powerful feature significantly simplifies the process of training neural networks.

**How it Works:**

1. **Computational Graph:** When you perform operations on PyTorch tensors, a computational graph is created. This graph represents the sequence of operations involved in calculating the output.
2. **Forward Pass:** During the forward pass, the input data is fed into the graph, and calculations are performed to obtain the output.
3. **Backward Pass:** In the backward pass, the gradients of the output with respect to the input parameters are computed using the chain rule. PyTorch automatically traverses the computational graph in reverse order, calculating gradients for each node.
4. **Gradient Update:** The calculated gradients are then used to update the model's parameters using an optimization algorithm like stochastic gradient descent (SGD).

**Key Benefits of PyTorch's Autodiff:**

- **Simplified Training:** Autodiff eliminates the need for manual gradient calculation, making the training process more efficient.
- **Complex Models:** It enables the training of complex deep learning models with numerous layers and parameters.
- **Flexibility:** PyTorch's dynamic computational graph allows for more flexibility in defining and training neural networks.

By understanding how PyTorch's autodiff works, you can effectively leverage this powerful tool to build and train sophisticated deep learning models.


[reference repo](https://github.com/yzh119/UW-dlsys-assignment1/blob/master/autodiff.py)


This test code gives good example about how to use autodiff to do forward and gradient calculation.

First we define variable in the graph.

We do some calculations with the node in the graph.
And then we first call `gradient()` to get gradient we are interested to get.
And then we call `Excutor()`  to create a instance of Executor to do forward pass
to get the value want.

In the constructor arg we give all the variables in graph we are interested to get value of
when `run()` is called. 

To actually run and get values we need to call `executor.run()` by giving values for all inputs.

```python
def test_matmul_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = ad.matmul_op(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]]) # 2x3

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)
```
