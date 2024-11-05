---
layout: post
title:  Stf CS149 Parallel Programming - Lecture 5&6 - Performance optimization
date: 2024-10-26 07:59:00-0400
description:  
tags:  parallel programming  
categories: parallel programming
featured: false
---




## Lecture 5
[Video](https://youtu.be/mmO2Ri_dJkk?si=CCG3Tf9dDYZiExq6)
Deciding granularity is important for dynamic scheduling in 
parallel programming.

Small granularity leads to better workload distribution
but comes with higher synchronization overhead.


## Lecture 6
Performance optimization: locality, communication and contention.

Reduce costs of communication between:
1. processors.
2. between processors and memory.


Shared memory communication.
Numa: non-uniform memory access


Message passing
blocking send and non-blocking send

Reduce communication is important to achieve max 
utilization of cpu. Just to keep cpu busy


Roofline model:
![image](https://github.com/user-attachments/assets/da01cc6b-a009-4b49-a306-c72940b89eaf)

To achieve maximum computation throughput GFLOPS/s
of cpu or gpus one has to have algorithm that has high
operation intensity -> high flops/bytes.


Need to has many computation per byte access unit.


