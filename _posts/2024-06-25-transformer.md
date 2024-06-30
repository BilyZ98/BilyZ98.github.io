---
layout: post
title: nano-gpt and Transformer  
date: 2024-06-28 07:59:00-0400
description: llm 
tags:  ml ai llm
categories: ml
featured: false
---

### Vanilla bigram model without self attention 
As mentioned in the youtube video [https://youtu.be/kCc8FmEb1nY?t=2509](https://youtu.be/kCc8FmEb1nY?t=2509),
this code builds a bigram model without self attention.
We can use this as baseline to compare with self attention code .

Issue: can not use torch cuda module even though I have gpu and install cuda pytorch
Solution: Tried to install pytorch cuda again
Get this error when I tried to install pytroch-cuda:12.1
```
ClobberError: This transaction has incompatible packages due to a shared path.
  packages: https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64::jpeg-9e-h5eee18b_1, pytorch/linux-64::libjpeg-turbo-2.0.0-h9bf148f_0
  path: 'share/man/man1/rdjpgcom.1'


ClobberError: This transaction has incompatible packages due to a shared path.
  packages: https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64::jpeg-9e-h5eee18b_1, pytorch/linux-64::libjpeg-turbo-2.0.0-h9bf148f_0
  path: 'share/man/man1/wrjpgcom.1'


```

Solution: Switch to new environment and reinstall pytorch with cuda

Comparison of cpu and gpu for bigram model
CPU:

GPU:



## Transformer architecture


