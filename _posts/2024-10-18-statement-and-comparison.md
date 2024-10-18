---
layout: post
title: C compiler - parse example walkthrough
date: 2024-10-14 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---




## Abstract syntax tree generation example 
```
"5+20-4;"
```

```asm
  .globl main
main: 
 mov $4, %rax
 push %rax
 mov $20, %rax
 push %rax
 mov $5, %rax
 pop %rdi
 add %rdi, %rax
 pop %rdi
 sub %rdi, %rax
  ret
```


The parse tree looks like this 
    
```
     ─  
   /  \ 
  +    4
/  \    
5  20
```

```cpp
static void gen_expr(Node* node) {
  switch(node->kind) {
    case ND_NUM:
    printf(" mov $%d, %%rax\n", node->val);
    return;

    case ND_NEG:
    gen_expr(node->lhs);
    printf(" neg %%rax\n");
    return;
  }

  gen_expr(node->rhs);
  push();
  gen_expr(node->lhs);
  pop("%rdi");

  switch(node->kind) {
    case ND_ADD:
    printf(" add %%rdi, %%rax\n");
    return;
    
    case ND_SUB:
    printf(" sub %%rdi, %%rax\n");
    return;


```

