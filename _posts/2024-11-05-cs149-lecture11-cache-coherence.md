---
layout: post
title:  Stf CS149 Parallel Programming - Lecture11 - Cache coherence
date: 2024-11-01 07:59:00-0400
description:  
tags:  parallel programming  
categories: parallel programming
featured: false
---


## Lecture 11: Cache coherence


Cache takes large amount of area in the chip

Cache bring performance boosting.

Two locality of data access pattern so that cache can help with 
performance.
1. Spatial locality
2. Temporal locality

Cache and cacheline in cpu.
Usually a single cacheline contains multiple
cache bytes(64bytes which can store 8 int for example ).

What is the problem with a shared cache processor design?

The scalability is the problem.

The cache bus has bandwidth limit..

A shared cache avoid the cache coherence problem but each write to a 
memory address will be broadcasted to other cores which waste bandwidth of bus.

Shared cache contention example:

Imagine you have a multi-core processor where two cores, Core 0 and Core 1, share the same Last Level Cache (LLC). If both cores are running different applications that frequently access and modify data stored in the shared cache, they will compete for the cache's resources. This competition can lead to contention.

For instance, consider two applications, App A and App B, running on Core 0 and Core 1, respectively. Both applications need to access large datasets that do not fit entirely in the cache. As App A accesses its data, it loads cache lines into the LLC, potentially evicting cache lines that App B needs. When App B tries to access its data, it may find that the required cache lines have been evicted by App A, causing cache misses and forcing App B to fetch the data from the slower main memory¹.

This back-and-forth eviction and reloading of cache lines between the two applications degrade their performance compared to a scenario where each application has its own private cache¹.


(2) 250P: Computer Systems Architecture Lecture 10: Caches - University of Utah. https://users.cs.utah.edu/~aburtsev/250P/2019fall/lectures/lecture10-caches/lecture10-caches.pdf.


### constructive and destructive interference
In the context of CPU caches, **constructive interference** and **destructive interference** refer to the effects of multiple processors accessing shared cache lines.

### Constructive Interference
Constructive interference occurs when multiple processors access the same data in a shared cache, leading to improved performance. For example, if one processor loads data into the shared cache that another processor also needs, the second processor can access this data quickly without having to fetch it from the slower main memory. This reduces cache misses and improves overall efficiency².

### Destructive Interference
Destructive interference, on the other hand, happens when multiple processors access different data that map to the same cache line, causing conflicts. This can lead to frequent cache line invalidations and reloads, increasing the number of cache misses and degrading performance. For instance, if two processors continuously overwrite each other's data in the same cache line, they will experience higher latency due to the constant need to fetch data from the main memory².

These concepts are crucial in designing efficient cache systems for multi-core processors, as they highlight the trade-offs between shared and private caches and the importance of cache coherence protocols.



