---
layout: post
title: nanogpt kv cache first attempt  
date: 2025-01-18 07:59:00-0400
description:  
tags:   ml llm 
categories: ml llm
featured: false
---



## 1. Run basic nano-gpt

```
git clone https://github.com/karpathy/nanoGPT.git

```

Install necessary packages
```
pip install -r requirements.txt
```

I have these packages in the requirements.txt
```
blobfile==2.0.1
certifi==2022.12.7
charset-normalizer==3.0.1
filelock==3.9.0
idna==3.4
lxml==4.9.2
numpy==1.24.2
pycryptodomex==3.17
pytz==2022.7.1
regex==2022.10.31
requests==2.28.2
tokenizers==0.13.2
torch==2.0.0
typing_extensions==4.4.0
urllib3==1.26.14
torch==2.0.0
numpy==1.24.2
transformers==4.28.1
datasets==2.11.0
tiktoken==0.3.3
wandb==0.14.2
tqdm==4.65.0
```


Follow quick start guidance in nanogpt repo do make sure that we 
can run training and inference successfully.
```
python data/shakespeare_char/prepare.py
python train.py --compile=False config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

My python version is 3.11 which is too high for model compile so I added
`--compile=False` in train command.


With my A800 gpu, I get a loss 0.0449 after 5000 iteration training.
```
iter 4970: loss 0.0461, time 18.12ms, mfu 20.21%
iter 4980: loss 0.0441, time 18.14ms, mfu 20.24%
iter 4990: loss 0.0464, time 18.13ms, mfu 20.27%
step 5000: train loss 0.0383, val loss 4.7262
iter 5000: loss 0.0449, time 3352.84ms, mfu 18.26%
```



## 2. Load GPT-2 models  checkpoints and test performance

https://stackoverflow.com/questions/75110981/sslerror-httpsconnectionpoolhost-huggingface-co-port-443-max-retries-exce


proxy error while trying to download gpt2 model from huggingface:
[https://github.com/huggingface/transformers/issues/17611](https://github.com/huggingface/transformers/issues/17611)

First downgrad requests version to 2.27.1
```bash
pip install requests==2.27.1
```
And then adding these two lines of code in `train.py` and `sample.py` fix the proxy connection issue for me
```python
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
```

Run `sample.py` to get a test of gpt2 model with params downloaded from huggingface.
```
 python sample.py --init_from='gpt2'
```

I tried to start with "please tell me a joke." The output is not anything like joke
but still very readable.
```
please tell me a joke

[…]

My name is Zarek, but I am extremely sad for you.

You can't even come to my house anymore

I'm sorry, I know

I have a dream

I don't know how long this thing will last

My name Is Zarek

I'm an adult who believes that

The problem with your friend is that he doesnt know

He doesn't know how to act
```


running time for 10 times inference:
```
---------------
Elapsed time: 25.4s
```

## 3. Implement KV cache for faster inference

[Commit hisotry for kv cache implementation](https://github.com/BilyZ98/nano-gpt-kv-cache/commit/606e4e4e881db6c769e0bdca51bdac96f00a55e1)

Please check code above for implementation details.

Issue:
```
shape of past k proj is  torch.Size([1, 12, 946, 64])
shape of k is  torch.Size([1, 12, 44, 64]) shape of v is  torch.Size([1, 12, 44, 64])
q len is  45
shape of past k proj is  torch.Size([1, 12, 990, 64])
shape of k is  torch.Size([1, 12, 45, 64]) shape of v is  torch.Size([1, 12, 45, 64])
Traceback (most recent call last):
  File "/GPUFS/nsccgz_qylin_1/zt/nano-gpt-kv-cache/sample.py", line 93, in <module>
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/miniconda3/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/zt/nano-gpt-kv-cache/model.py", line 359, in generate
    logits, _, past_kv_proj = self(idx_cond, past_kv_proj=past_kv_proj,start_pos=start_pos)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/miniconda3/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/zt/nano-gpt-kv-cache/model.py", line 204, in forward
    x, layer_kv_proj = block(x, past_kv_proj=past_kv_proj[i])
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/miniconda3/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/zt/nano-gpt-kv-cache/model.py", line 122, in forward
    attn_res, present_kv_proj = self.attn(self.ln_1(x), past_kv_proj=past_kv_proj)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/miniconda3/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/GPUFS/nsccgz_qylin_1/zt/nano-gpt-kv-cache/model.py", line 78, in forward
    assert KV < self.block_size, f"KV: {KV} >= block_size: {self.block_size}"
           ^^^^^^^^^^^^^^^^^^^^
AssertionError: KV: 1035 >= block_size: 1024
yhrun: error: gpu73: task 0: Exited with exit code 1
(nano-gpt-kv-cache) [nsccgz_qylin_1@ln101 nano-gpt-kv-cache]$
```


Fix
```python
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # This is the righ condition
            if idx.size(1)  == T:
                idx_cond = idx
                start_pos = 0
            else:
                idx_cond = idx[:, [-1]]
                start_pos = idx.size(1) - 1
```

The limitation of this code is that it can only handles 
condition where `max_new_tokens < self.config.block_size`


I don't know why yet.
## 4. Test KV cache performance

[The commit](https://github.com/karpathy/nanoGPT/pull/76) mentions that it only brings performance 
boost with cpu but not on A100 gpu. Why is that ? 
Is this because that linear projections can be quickly done with
fast gpu matrix multiplication?


[This commit](https://github.com/huggingface/transformers/pull/14118/files) 
[and this discussion](https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227)
talks about how to handle long text generation. I have not yet 
understanded it completely how it deals with long text geneartion.

There is a technique called rotary positional embeddings as mentioned in this 
[commit](https://github.com/karpathy/nanoGPT/pull/76).
But I don't know how does it works yet. And all I want to do right now is to
simply test how kv cache helps with inference speed.


My naive solution right now is to simply cut 
past_kv_proj to latest self.config.block_size tokens
```
            if past_kv_proj is not None:
                past_k_proj, past_v_proj = past_kv_proj
                print('shape of past k proj is ', past_k_proj.shape )
                print('shape of k is ', k.shape, 'shape of v is ', v.shape)
                if KV >= self.block_size:
                    past_k_proj = past_k_proj[:, :, -self.block_size:, :]
                    past_v_proj = past_v_proj[:, :, -self.block_size:, :]
                k = torch.cat((past_k_proj, k), dim=2)
                v = torch.cat((past_v_proj, v), dim=2)

```


gpu v100

with kv cache, no flash attention

```
 yhrun -p gpu_v100  python   sample.py --init_from='gpt2'  --use_kv_cache=True --dtype=float32  --num_samples=10 --max_
new_tokens=1000
```
time:
```
---------------
Elapsed time: 102.6s
```
memory:


without kv cache, no flash attention
```
python   sample.py --init_from='gpt2'  --use_kv_cache=False --dtype=float32  --num_samples=10 --max_new_tokens=1000
```
time:
```
Elapsed time: 151.8s
```
memory:


Saves 30% time. Not bad.



500 tokens, cpu

with kv cache, 
```
The law gives the government access to consumer information only if the government's purpose is to provide health care to the general public. If those
---------------
Elapsed time: 218.9s


The law gives the government access to consumer information only if the government's purpose is to provide health care to the general public. If those
---------------
Elapsed time: 251.4s
```


without kv cache
```
The law gives the government access to consumer information only if the government's purpose is to provide health care to the general public. If those
---------------
Elapsed time: 1191.4s
```

5 times inference time saving. Not bad.


The peak memory usage between with kv cache and without kv cache is nearly the same.
This is because that sequence length is the same with or without kv cache.
However, kv cache do bring some advantages. Here's the answer from gpt.


> Actually, there is a difference in memory usage when using KV cache for LLM inference. While it's true that the maximum memory usage might be similar, the way memory is utilized and managed can vary significantly.
> 1. **Memory Allocation**: With KV cache, memory is allocated for storing key-value pairs from previous computations. This can lead to more efficient memory usage as the model doesn't need to recompute values, reducing the overall memory footprint during inference.
> 2. **Memory Management**: KV cache helps in better memory management by reusing previously computed values. This can lead to more stable memory usage patterns, avoiding spikes in memory consumption that might occur without caching.
> 3. **Performance Optimization**: By reducing redundant computations, KV cache can lead to faster inference times, which indirectly affects memory usage. Faster computations mean less time spent holding intermediate values in memory, leading to more efficient memory utilization.

<!-- lol. It does not improves any. Why is that? -->
<!-- There is a bug in my code that does not feed all  -->
<!-- previously generated tokens into model when `use_kv_cache=False` -->


## References

[youtube video llm kv cache explanation](https://www.youtube.com/watch?v=80bIUggRJf4&t=247s)

[requirements.txt to run nano-gpt](https://github.com/karpathy/nanoGPT/pull/246/commits/5cc9bab7e2402caf69a00e9c38fc45517e958748)

[nano-gpt kv cache pr example](https://github.com/karpathy/nanoGPT/pull/76)

[huggingface transformers kv cache source code on github](https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/modeling_outputs.py#L714)

https://zhuanlan.zhihu.com/p/646577898

https://zhuanlan.zhihu.com/p/624740065

[huggingface transformers API documentation](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast)




