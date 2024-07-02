---
layout: post
title: nano-gpt and Transformer  
date: 2024-06-29 07:59:00-0400
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
Transformer is a type of neural network architecture that is designed to handle sequential data more effectively than traditional RNNs and LSTMs. 
It was introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017, and has since become a popular choice for natural language processing tasks.

As mentioned in this video 
transformer is like neural network version of map-reduce which also 
comes from google.
The reduce process is the self attention process in transformer. 
The map process is the feed forward neural network and mutl-head in transformer.

Why does transformer have Feedforward and Linear at the same time ? These two looks like the same.
<img src="https://github.com/BilyZ98/BilyZ98.github.io/assets/26542149/4ce5458c-3c90-4607-803c-01631327ad0f" width="500" height="500">


### bigram model with cpu

Code link: [https://github.com/BilyZ98/nano-gpt/blob/5cae2e1635dc560dc75dc92897ee5add43fa3aed/bigram.py](https://github.com/BilyZ98/nano-gpt/blob/5cae2e1635dc560dc75dc92897ee5add43fa3aed/bigram.py)
```
step <built-in function iter>: train loss2.5979, val loss 2.5999
Time taken: 38.3419029712677 seconds

SOd HADghe uio choaxlondorerl cy,m thamaw pes$!
LUANIV:
R:
RYo,
W:'th thiveCond wnff tod ghind.
Thathis:Ved esVI!
RUSape ms, yonyail lomustthend? thed of sofatiatherves f het m ssprerh fon,cke d pr&lR.
-IUq-bind p'y w; deland walois WBy ethu l'd t y montircPEMPlanas y dslly?sthan coor ccoust d limald ped il f frs th.
ce our ntLE:

YCEqusI,
K:
wnea!vengjF, 'd
GGe cltLTod.
RSwoppiQYe haland dSt tXESacedDUCORGRENETof hos, sooumouloo meRTooe,
The cke;
ONGu he tpalapy an:
NScheracancoj

HAnend
ANUARK
```

### bigram model with gpu with positional embedding and language model head
Issue:
Get this error while running on gpu
```
/Indexing.cu:1236: indexSelectSmallIndex: block: [0,0,0], thread: [1,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
```

Searched google and it tells that the problem comes from incorrect indexing while using nn.Embedding.

Solution: 
Check nn.Embedding code and fix it.
Fix issue above by adding following code to `generate` function.
```python
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] # This line of code fix the issue
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :] # becomes (B, C)
      probs = F.softmax(logits, dim=-1) # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
      idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
    return idx
```
The  reason is that `self.token_embedding_table` and `self.position_embedding_table` 
shares different input dimensions.

If we don't crop the `idx` to `idx_cond`, the `pos_emb`
will take `T` that is larger than `block_size` which will cause the error.

Please check this time in the video [https://youtu.be/kCc8FmEb1nY?t=4854](https://youtu.be/kCc8FmEb1nY?t=4854)



