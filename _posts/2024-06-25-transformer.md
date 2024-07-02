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
No self attention for this version

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

code:
```python
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    # self.feed_forward = nn.Linear()

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)


    loss = None
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T) if targets is not None else None
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      #print('shape of logits', logits.shape)
      logits = logits[:, -1, :] # becomes (B, C)
      probs = F.softmax(logits, dim=-1) # (B, C)
      idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
      idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
    return idx
```

Output:
```
model device cuda:0
m device cuda:0
step <built-in function iter>: train loss2.4926, val loss 2.5021
Time taken: 28.21234440803528 seconds



CEThik bridcowindakis by ble

Hiset bobe d e.
S:
O:
ISM:


Thanss:
Wanthar u qur, vet?
F dilasoate awice my.

Hnstarom oroup
Yowhthetof isth ble mil ndilll,

W:

Yeesengcin lat Heriliov ts, and Win nghire yombousel lind pe llllishe ce hiry:
Supr aisspllw y.
Hllin's noroopetelaves
Momy ll, d mothakeeo W-ndo whthCeiibyo touth dourive weeshieed t so mower; te

AN ad nterupt f s ar iris! m:
```


### Self attention with single head on gpu with positional embedding and language model head
Code:
```python
class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, C)
        q = self.query(x) #(B, T, C)
        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        v = self.value(x) #(B, T, C)
        out = wei @ v #(B,T,T) @ ( B, T, C) -> (B, T, C)
        return out


class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.sa = Head(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    # self.feed_forward = nn.Linear()

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    x = self.sa(x)  #(B,T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)


    loss = None
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T) if targets is not None else None
      loss = F.cross_entropy(logits, targets)

    return logits, loss
```
Output:
```
step <built-in function iter>: train loss2.4015, val loss 2.4166
Time taken: 56.774349212646484 seconds

Whent ik bry cowilen is by bth

Hiset bobe ale.
S:
O:
IS:
Falilauss ar btharu wearthe.
War dilasoate awice my.

HDER:
ANGo oug
Yowhavetof is he ot mil; dill, aes iree sen cie lat Herid ovets, and Win ngar ilerabous lelind peal.
-hull onchiry ptugr aiss hew ye wllinde norod atelaves
Momy ll, dl othake ont---o whth eiiby we ati dourive wee, ired thoouso er; th
To kad nteruptef so;
ARID Wam:
ENGCI inleront ffaf Pre?
```

It does not improve a lot.



### Multi-head self attention on gpu with positional embedding and language model head
Code:
```python

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa = Head(n_embd)
    self.sa_heads = MultiHeadAttention(4, n_embd//4)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    # self.feed_forward = nn.Linear()

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    # x = self.sa(x)  #(B,T, C)
    x = self.sa_heads(x)
    logits = self.lm_head(x) # (B, T, vocab_size)


    loss = None
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T) if targets is not None else None
      loss = F.cross_entropy(logits, targets)

    return logits, loss
```


```
step <built-in function iter>: train loss2.2748, val loss 2.2858
Time taken: 90.66784453392029 seconds

Whent if bridcowd, whis byer that set bobe toe anthr-and mealleands:
Warth foulque, vet?
Wedtlay anes wice my.

HDY'n om oroug
Yowns, tof is heir thil; dill, aes isee sen cin lat Hetilrov the and Win now onderabousel.

SFAUS:
Shenser cechiry prugh aissthe, ye wing, u not
To thig I whomeny wod mothake ont---An hat evibys wietit, stile weeshirecs poor gier; to
To k danteref If sor; igre! mef thre inledo the af Pre?

WISo myay I sup!
Atied is:
Sadsal the E'd st hoin couk aar tey Iry to I frouf voul
```

It looks better and the loss continues to decrease.
But it takes longer to finish. Why is that ? 


### Multi-head self attention with feed forward neural network on gpu with positional embedding and language model head

Why don't we add positional embedding again between blocks ?  
I think this can help transformer to keep track of the position of the tokens.

I think we don't need to add positional embedding again and again between blocks once we 
use residual connection.

Code:
```python
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa = Head(n_embd)
    self.sa_heads = MultiHeadAttention(4, n_embd//4)
    self.ffw = FeedForward(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    # self.feed_forward = nn.Linear()

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    # x = self.sa(x)  #(B,T, C)
    x = self.sa_heads(x) #(B, T, C)
    x = self.ffw(x) #(B, T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)


    loss = None
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T) if targets is not None else None
      loss = F.cross_entropy(logits, targets)

    return logits, loss
```



output:
```
step <built-in function iter>: train loss2.2288, val loss 2.2414
Time taken: 102.24195170402527 seconds

And the Ror
Thow and is and thrad thom of oule.
Sthr-' my dall ands:
Warth fou qurord.
War dilth ane aw crup and not, ut onour
Yowns, tof it he cove lend lincath is ees, hain lat Het dulvets, and to poman is wables lill dite ullliser cecrivy prupthaiss hew youn's and knamopetell lownomthy wod moth keacal---A wher eiicks to thour rive cees, meds pood of he thu the hanterth po so;; igis! my to thy ale ontat af Pried my of.
WHINY ICHARD:
Poid:
Ardsal the Eget to uin cour ay andy Rry to chan the!
An
```

Loss continue s to decrease but not decreases a lot



### Blocks of multi-head self attention on gpu

Code:
```python
class Block(nn.Module):
    """Transfomer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd )

    def forward(self, x):
        x = self.sa(x)
        x = self.ffw(x)
        return x

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa = Head(n_embd)
    # self.sa_heads = MultiHeadAttention(4, n_embd//4)
    # self.ffw = FeedForward(n_embd)
    self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
    )
    self.lm_head = nn.Linear(n_embd, vocab_size)
    # self.feed_forward = nn.Linear()

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    # x = self.sa(x)  #(B,T, C)
    # x = self.sa_heads(x) #(B, T, C)
    # x = self.ffw(x) #(B, T, C)
    x = self.blocks(x)
    logits = self.lm_head(x) # (B, T, vocab_size)


    loss = None
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T) if targets is not None else None
      loss = F.cross_entropy(logits, targets)

    return logits, loss
```
```
step <built-in function iter>: train loss2.3255, val loss 2.3400
Time taken: 141.53945779800415 seconds

And thif bry cowh, har on, ber
waiset bobe to tavegr-d my dalceauss:
Want he us he hentbardethas ane awche my.

HDEE:
Ay neou waowns
Moof is he me mil; dill, aes ireees, hain latiser drovets, and the nor ond wabousel lind thau.
Hhult cncriby: thartaiss hew you lome.
I yof petelgolg's my yow demeth kleonW nou when eiibas wouth dotrive weeshime sto-oche eroure
Thak danterurt fou ar irist muf thin inle oft to fearr?

KISomerry youu
Hartied is:
Aadsalce.

EIDLHY:
Iin couk aaraney Iry the han yo vely
```

Does not improve
### Blocks of multi-head self attention with residule connection on gpu
No projecttions for residual connection:
Code:

I only show the code for `Block` class because this is the only change in the code.
```python
class Block(nn.Module):
    """Transfomer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd )

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffw(x)
        return x
```

Output:
```
step <built-in function iter>: train loss2.1115, val loss 2.1749
Time taken: 155.8028919696808 seconds

And they bridcown,
The layest madisen bube to tamaght' my daliea
My:
Waith foulqorth
but ceetlay ane awice my.

HEER:
An onour
Yount
Moofuing come mill dill, at miree seng, wilatist in ove the Bent longht is wais welll no me litles;
So chirs: ther aiss haw youn's mause roodeter'd swer:
Ill o' meacke
Ao Windo wht Ceiiby we ath do rive wees ire sto-of of he the the danterty po so;
Ang hink:
'Elt yould ontates
Mare?

KING ENCHENNL:
Hartied is wards beaces and thisin cour ay and
Hire the have fove y
```

With projecttions for residual connection:

Code:
```python

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)


    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd, n_embd*4),
                nn.ReLU(),
                nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transfomer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd )

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffw(x)
        return x
```

Output:
```
step <built-in function iter>: train loss2.0052, val loss 2.1047
Time taken: 128.87973642349243 seconds


KER:
Dy be will and is by be madised bube to take Our my dagalanss:
Wact me us crome. Wardethas anes wick, you, not to zoknow
Yourselvef is heart milled,
What grive, send, will is therevers, and the now on you me, lord dime littishe courmby pruperais'll woy. Hurmake norfore blaves home.
Who my thake of in on her eis as the most rive cenchimed the come, for unter hands thime son; if hink:
Edway male of wefife
Where, Som.
What suk!
Kered is wards.
Wice Efees bidin couses.
Wher, reath chan the wel
```

It's a little bit better compared to no-projection.

What is the difference between these two ?
 I think projection is used to project output tensor from 
self attention to the same space with input tensor so that we can add them together.

But from the result I can see the not projecting is also fine.



### Residual blocks of self attention with layernorm

Code:
```python
class Block(nn.Module):
    """Transfomer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
    )

    self.lm_head = nn.Linear(n_embd, vocab_size)
    # self.feed_forward = nn.Linear()

```

Output:
```
step <built-in function iter>: train loss1.9904, val loss 2.0918
Time taken: 165.2521140575409 seconds

And they bridle.

SOROR To beer a seek obe don.
Sagrad my dagalans!
You that us hear buble dilt
Hate away, my fears'd of of my
Yoursert foitie bettlit now
Whimes if ensen cim;
Stistaid ove the the me now on that thell in a wall thus would by pruppiness hiw you:
That I mandpeter'd gond:
Is would that
To Winson her eis all'd they srive will ime strow more-fore
To knom thrupt for trear. Wame monge inlee,
Thef firse?

KISTINUS:
If be!

GRESNY:

Sadave the Edwall?

GRAKE Masceave
Hir-bromence you! My
```

loss drops a little bit more compared to no layernorm.

### Residual blocks of self attention with layernorm + dropout (Full transformer)
```python
class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, C)
        q = self.query(x) #(B, T, C)
        wei = q @ k.transpose(-2, -1) * C **-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) #(B, T, C)
        out = wei @ v #(B,T,T) @ ( B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd, n_embd*4),
                nn.ReLU(),
                nn.Linear(4*n_embd, n_embd),
                nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

```


Output:
```
step <built-in function iter>: train loss2.1027, val loss 2.1580
Time taken: 155.4465034008026 seconds


LONIBENT:
Be a most unre maim, gruch, thath pcourdith my it frilf oblis then, nrie?

Mor hould bace                                                                                                                                                          Ce in apitng, anoth you his to cowll
By mbre wand grarist let fead as be meest, Jo afore by shalve my my sade make ta gior mony ow norane;                                                                   Hould-wrind awnAndead notooth. WARKEIY:                                                                                                                                                                                                                                                                                                         Conear gy?                                                                                                                                                              Srom ands, his gahpe with gowis slined fue no lot all wopmeseond in he tha dee knoth quail hen, slyold aus mawers, slosssig, yat but, hery,                             Ond you hom is oalt in, shealve of dRulet my bafker's deforth the sh
```
It's not getting better.
I think this dropout will help when we scale up number of parameters.
### Full transfomer with more parameters
Previous parameters
```python

batch_size = 32
block_size = 8 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('torch cuda available', torch.cuda.is_available())
eval_iters = 200
n_embd = 32
head_size = 16
dropout = 0.2
```


Cur param:

Output:
```
step <built-in function iter>: train loss1.8683, val loss 2.0060
Time taken: 221.6930215358734 seconds

Bale herse?

LAURENNIUS:
When regoubjon theme as be boeet chal speak treatuls a faulse heave ticeso too out boons in of, privy; char dese in paindaur meet is your both talies so are furnt ereworry,
Besse do you grait see fiee in vile should but roonouth
Than I k ount crow on
Sint, I eid, doust not will comon't te refore wife, which young so hing me the grow by-treate, witee of sword That the we great his worse a mick's sestit well sue
I inn frie dam west you that,S more think
That yest deart com
```
loss does not drop to 1.4 which the output in karparthy's video and the running time 
is too short.

I see, there is a duplicate definition of `batch_size=4` and `block_size=8` in code.
Let's try to run again.
### Full transformer without positional embedding



