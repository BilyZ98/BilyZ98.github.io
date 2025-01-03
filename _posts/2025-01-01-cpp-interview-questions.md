---
layout: post
title: C++ and linux kernel memory allocation
date: 2025-01-01 07:59:00-0400
description:  
tags:  c compiler cpp linux  
categories: compiler
featured: false
---





[How malloc allocates memory?](https://mp.weixin.qq.com/s/EIVDCWHr8EFEl4LX6UqWHQ)



kernel memory allocation

- buddy algorithm

- slab allocator

The slab allocator and buddy algorithm are two different memory allocation techniques used in the Linux kernel, each with its own strengths and use cases.

### Buddy Algorithm
The buddy algorithm is used for allocating memory in large contiguous blocks. It works by dividing memory into blocks of size 2^n and maintaining a binary tree structure. When a memory request is made, the allocator finds the smallest block that can satisfy the request. If a larger block is found, it is split into smaller blocks (buddies) until the desired size is obtained. This method is efficient for allocating large blocks of memory and helps reduce fragmentation](https://www.tutorialspoint.com/allocating-kernel-memory-buddy-system-and-slab-system).

### Slab Allocator
The slab allocator, on the other hand, is designed for allocating small, frequently used objects. It maintains caches of pre-allocated memory chunks (slabs) for different object sizes. When an object is needed, it is allocated from the appropriate slab, which reduces the overhead of frequent allocations and deallocations. This method is particularly useful for kernel objects like process descriptors, file descriptors, and other small data structures](https://www.tutorialspoint.com/allocating-kernel-memory-buddy-system-and-slab-system).

### Relationship
The relationship between the two lies in their complementary roles within the Linux kernel. The buddy algorithm is used for allocating large contiguous blocks of memory, which can then be subdivided by the slab allocator for smaller, frequently used objects. This combination allows the Linux kernel to efficiently manage memory for both large and small allocations, optimizing performance and reducing fragmentation](https://www.tutorialspoint.com/allocating-kernel-memory-buddy-system-and-slab-system).


[ref 1](https://www.tutorialspoint.com/allocating-kernel-memory-buddy-system-and-slab-system)


### which algorithm does linux use when brk() syscall is called to allocate memory ?
When the `brk()` system call is used in Linux to allocate memory, it primarily relies on the **buddy algorithm** for managing the memory. The `brk()` system call adjusts the end of the data segment of a process, effectively increasing or decreasing the amount of memory allocated to the process. The buddy algorithm is used to allocate large contiguous blocks of memory, which can then be managed by the process.


### Difference between malloc and new in c++ ?


malloc is a system call function. When calling malloc it will not call constructor of
class or do memory initialization. `free` to free memory.

new is a primitive word in c++ itself. `new` will call constructor of class and do 
memory initialization. `new` is managed by c++ compiler. `delete` to free memory and destructor 
is called for `delete`



### Memory allocation for process in linux
1. For small memory allocation `brk` is called.

    `brk` will allocate memory from heap and the allocated memory will not be returned 
    to kernel immediately after `free`. It will stay in the process memory mapping space
    for future use.

2. For large memory allocation `mmap` is called.
    For memory size allocation request > 128KB, memory allocated is in file mapping 
    space in process memory space. The physical memory is returned to kernel immediately
    after `free` call which is different from `brk`


[ref article about memory allocation ](https://mp.weixin.qq.com/s/EIVDCWHr8EFEl4LX6UqWHQ)
