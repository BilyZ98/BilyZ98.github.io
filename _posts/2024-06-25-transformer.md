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
[https://pytorch.org/get-started/locally/#windows-anaconda](https://pytorch.org/get-started/locally/#windows-anaconda)
```
conda clean --all
conda clean -p
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Now I am able to see cuda available in pytorch
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # 0 corresponds to the first GPU
```

```
nano-gpt
True
NVIDIA A800 80GB PCIe
```



Comparison of cpu and gpu for bigram model
CPU:

Time taken: 14.610548734664917 seconds

GPU:

Time taken: 18.080146074295044 seconds

It takes longer time for gpu to finish. 
I think it's because training iteration is not large enough to see the benefit of gpu.


## Transformer architecture
Why does transformer have Feedforward and Linear at the same time ? These two looks like the same.
<img src="https://github.com/BilyZ98/BilyZ98.github.io/assets/26542149/4ce5458c-3c90-4607-803c-01631327ad0f" width="500" height="500">

