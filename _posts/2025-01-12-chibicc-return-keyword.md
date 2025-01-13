---
layout: post
title: chibicc - Simple c compiler return keyword  
date: 2025-01-12 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---

## Add return keyword to simple c compiler


[Commit history](https://github.com/BilyZ98/chibicc/commit/f94ca394ade26ca861bd205d3714f103eb4dedb9)

For tokenizer, `convert_keywords()` is added to convert kind of token
from identity to keyword.
So this means that all basic tokens are identity at first and later
convert to keywrod type token.


For parser, add extra prudction rule/grammar rule in expr geneartion.
`stmt = "return" expr ";" | expr-stmt`. 
New node type `ND_RETURN` is added.

For code generation, `jmp .L.return` is added to jump to specified assembly code.

For test, add `return 1; 2; 3` to test return actually works.



