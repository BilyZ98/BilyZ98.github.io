---
layout: post
title: BPE tokenizer implementation
date: 2025-02-15 07:59:00-0400
description:  
tags:   llm  tokenization gpt
categories:  ml
featured: false
---




## What is tokenization and why we need it?
tokenization is the proess of encoding original raw text
into a shorter string of representation.

For example,
for word "the" its binary represnetation is "[116,104, 101]"
which requires 3 numbers to store. With tokenziation we can merge 
these three bytes into one number as "256".

So basically, tokenization is the process of compression of the input text.
Why compress it? Because for transformers, input context length is limited 
and we want to put as much as information in the limited context window
to process as much as information in one time.
So tokenization helps us to train model to process longers input text.

## Byte-pair-encoding(BPE) tokenization algorithm

The basic ideas of bpe algorithm is to identify 
occuring adjacent pair of tokens and merge them by 
assigning a new token number for this pair of tokenso


For example,
suppose we have training text `the cat in the hat`,
in the text, "th" appear twice, so at the beinging of
the algorithm we replace the text like this `<256>e cat in <256>e hat`.
So we replace two words with one number aka. the text is shorten and compressed.

The vocabulary is 
```
 0: ...
  ...
  256: "th"
```

And we can repeat this process by merging `<256>e` ito `<257>` because 
`<256>e` occurs twice in the text.
The vocabulary is 
```
0: ...
...
256: "th"
257: "<256>e"
```

So from steps above we can see that more compressed the text is , the larger 
vocabulary size we get because we have more new tokens in the vocabulary.

Steps

1. Identify frequent pairs

2. Replace and record

3. Repeat until no gains.

code.

1. Identify requent pairs
```cpp
def get_stats(tokens):
  pair_count = {}
  for pair in zip(tokens, tokens[1:]):
    pair_count[pair] = pair_count.get(pair, 0) + 1

  return pair_count

stats = get_stats(tokens)
print(get_stats(tokens))
print("most occuring pair", max(stats, key=stats.get),  'occuring count', stats[max(stats, key=stats.get)])
top_pair = max(stats, key=stats.get)
```

2. Replace and record ( until reach maximum vocab size )
```python
def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# ---
vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx
```

### Encoding
Once we finishing training our tokenizer we can use that tokenizer to tokenize any new input texts

```python
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break

        idx = merges[pair]
        tokens = merge(tokens, pair, idx)

    return tokens

    
print(decode(encode("hello world")))


```

Using the `min` function with the custom key helps select the most suitable pair of tokens for merging 
based on defined criteria. Here's an example to illustrate its purpose:


Imagine you have a list of token pairs with their merge priorities stored in a dictionary `merges`. 
You want to find the pair with the lowest priority value that exists in this dictionary.

Consider the following example:

```python
merges = {('th', 'e'): 1, ('a', 'n'): 2, ('i', 'n'): 3, ('s', 't'): 4}
stats = {('th', 'e'): 5, ('a', 'n'): 3, ('i', 'n'): 2, ('h', 'e'): 4}
```

In this case, the `merges` dictionary contains pairs with their priorities, and the `stats` dictionary contains pairs with their frequencies. You want to find the pair with the lowest priority that exists in the `merges` dictionary.

The code `pair = min(stats, key=lambda p: merges.get(p, float("inf")))` will evaluate as follows:

1. For each pair in `stats`:
    - ('th', 'e'): `merges.get(('th', 'e'), float("inf"))` returns `1`
    - ('a', 'n'): `merges.get(('a', 'n'), float("inf"))` returns `2`
    - ('i', 'n'): `merges.get(('i', 'n'), float("inf"))` returns `3`
    - ('h', 'e'): `merges.get(('h', 'e'), float("inf"))` returns `float("inf")` (not in `merges`)

2. The `min` function selects the pair with the smallest value:
    - `('th', 'e')` with a priority of `1`

Therefore, `pair` will be `('th', 'e')`, the pair with the lowest priority for merging. This approach ensures that only pairs defined in the `merges` dictionary are considered and prioritizes them based on their defined values. 
This makes the merging process efficient and controlled according to specific criteria.

### Decoding
Since each new token(id >= 255) generated is represented by two sub tokens
we can just concat the sub tokens to build original new tokens.

```python
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

print(decode([128]))
```

## References

[tokenization notebook from karpathy](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing#scrollTo=ZU2Qwf-5Ohvn)

[gpt tokenzier video from karpathy](https://www.youtube.com/watch?v=zduSFxRajkE&t=2358s)

[bpe post from Sebastian Raschka](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
