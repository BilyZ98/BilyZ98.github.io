---
layout: post
title: Hash in cpp   
date: 2024-12-22 07:59:00-0400
description:  
tags:  algorithm  cpp  
categories: interview 
featured: false
---


## Open hashing and close hashing

What is open hashing and close hashing ?

### Open hashing (separate chaining)

Method: Each bucket in the table contains a linked list of all elements that hash to 
the same index

Collision handling: the new element is added to the list of the corresponding linked list(bucket)


Memory usage: requires additional memory for pointers in linked list

Performance: generally good compared to close hashing.

Example: `std::unordered_map` in cpp


### Close hashing (open addressing) 

Method: All elements are stored in the table itself. When a collision occurs
the algorithm searched for next available slot in the table

Collision handling: linear probing, quadratic probing, or double hashing.

Memory usage: less compared to open hashing

Performance: performance can degrade if the table is full, leading to longer search time

Example: python uses close hashing.

## Load factor in hash table

load factor is defined as number of elements to number of buckets in the table
$$
load factor  = \frac{n}{b}
$$


Example


If a hash table has 100 elements and 150 buckets, the load factor is 

$$
load factor  = \frac{100}{150} = 0.67
$$



## unordered_map in cpp
This article talks about internal implementation of `unordered_map` in cpp.

It's very intuitive.

[Explanation of internal implementation of unordered_map](https://jbseg.medium.com/c-unordered-map-under-the-hood-9540cec4553a)

[This stackoverflow post talks about why unordered_map in cpp uses open hashing](https://stackoverflow.com/questions/31112852/how-stdunordered-map-is-implemented)


