---
layout: post
title: micrograd   
date: 2024-08-05 07:59:00-0400
description:  
tags:  ml pytorch  
categories: ml
featured: false
---



What is micrograd?

micrograd is a simple implementation
of pytorch-like autograd engine built by karparthy.


## Test code
```py

from micrograd.engine import Value

a = Value(4.0)
b = Value(3.0)
c = a * b
print(f'{c.data:.4f}')

c.backward()
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')
print(f'{c.grad:.4f}')
```

The output is quite straightforward:
```
12.0000
3.0000
4.0000
1.0000
```

However, the grad will accumulate in the value if it's not set to zero.
```py
from micrograd.engine import Value

a = Value(4.0)
b = Value(3.0)
c = a * b
print(f'{c.data:.4f}')

c.backward()
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')
print(f'{c.grad:.4f}')


# a = Value(4.0)
# b = Value(3.0)
d = a + b
d.backward()
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')


```

OUtput:

```
12.0000
3.0000
4.0000
1.0000
4.0000
5.0000
```

### Internal implementation
Please check code in this notebook
[micrograd note book](https://colab.research.google.com/drive/1KF6houJ-X_uLIgZ5BaSV24-GnYpTQdeh?usp=sharing)
