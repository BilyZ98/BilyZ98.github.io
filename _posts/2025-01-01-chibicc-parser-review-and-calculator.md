---
layout: post
title: chibicc C compiler - parser review and expression evaluator
date: 2025-01-01 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---




## lea and mov explanation


`lea -4(%rbp), %rax` means that memory address that is 4 bytes below current base pointer 
`rbp` is stored in `rax` register.

For example,

assume `%rbp` holds value `0x7fffffffddd0` offset is 8,

then this code 
```
lea -8(%rbp), %rax
```
means 

1. the effective address calculation is `0x7fffffffddd0 -8 = 0x7fffffffdd8`
2. The value  `0x7fffffffdd8` is stored in `rax` register


I get confused about whether `mov src, dest` or `mov dest, src` is correct.

I learn from this [stack overflow post about mov in x86 and AT&T](https://stackoverflow.com/questions/5890724/mov-instruction-in-x86-assembly)
that both are valid.

`mov src, dest` is correct in AT&T and `mov dest, src` is valid in Intel syntax.



[Intel x86 mov explanation](https://electronicsreference.com/assembly-language/mov/)


### gcc uses AT&T assembly standard

GCC (GNU Compiler Collection) uses the AT&T assembly syntax by default. This is the standard assembly syntax used in Unix-like systems. However, GCC also supports Intel syntax, and you can switch to it using specific compiler flags if needed.

If you're working on a project and need to use Intel syntax, you can enable it with the `-masm=intel` flag. For example:

```bash
gcc -masm=intel -o myprogram myprogram.c
```


