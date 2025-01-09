---
layout: post
title: Learned idnex survey     
date: 2025-01-08 07:59:00-0400
description:  
tags:   learned index research
categories: learned index research 
featured: false
---




## What is learned index and why?
To search a key in b+ tree, traditional way is to do binary search.
This complexity time is O(Log_n)

Learned index build model to learn the distribution of keys in space.
The input is the key number and the output of the model is the location
of the key in the storage array space.
The time complexity is O(1) which is faster than binary search.


## Current work 

[The case for learned index structures](https://arxiv.org/pdf/1712.01208)
proposes to replace every node in b+ tree with learned model.

Each node contains a model trained from the keys covered by its key ranges.

It's said that in the paper that this hierachy style allows allows model
to learn the rough key distribtuion at its key ranges. Bottom level model 
covers smalles range of keys, it's easy for the model learn.
Top level covers larget range of keys but it only cares about the big structure
of the keys distribution and it leaves fine-grained key position to bottom level
models.


Search process: 

Model at each level gives its key position prediction until reaches the leaf child.

Since model might not give the correct position prediction, learned index make sures 
that each model prediction error is within a predefined error bound.

So it's guarantee that key is located within the [lower_bound, upper_bound] range 
given by the model.


This paper only mentions how to build model for read only scenario. 


[ALEX](https://arxiv.org/pdf/1905.08898) solves the updatable learned index problem.

How ?

Insert process:

For non-full data node, it inserts the key to the predicted position from the model
if there is a empty slow in the array..

The predcited position might be occupied, so it shifts other elements towards the closet
gap by one.  ( I am not sure if the model should be trained after this shift operation,
I think we should train a new model because that position of shifted key is changed,
but the model train cost is too high if we do this for each occupied key.)



For full-data node, it can choose to split the data node or matain a single data node but
with allocation of largers storage space and retarined a new model. 

For data node expansion, new position in array of original key is given by new model. 

This is not the same as traditional binary tree which just does simple copy.

For internal nodes, ALEX  choose to split the internal nodes horizontally, i.e at the 
same level.  


Alex can choose to turn data nodes to internal nodes which is the same as split the node
vertically, increasing the depth of the tree?

## What's next ? 

[https://arxiv.org/pdf/2403.06456]

## References

[Jeaf dean's talk about ml for sys in NIPS'25]()

[qd-tree](https://zongheng.me/pubs/qdtree-sigmod20.pdf) . Why only for data analytics? 


[SageDB: A Learned Database System](https://www.cidrdb.org/cidr2019/papers/p117-kraska-cidr19.pdf)


[Learned Index: A Comprehensive Experimental Evaluation](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/experiment-learned-index.pdf)

[Fast RDMA-based Ordered Key-Value Store using Remote Learned Cache](https://www.usenix.org/conference/osdi20/presentation/wei)
