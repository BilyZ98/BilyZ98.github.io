---
layout: post
title: chibicc - Simple c compiler unary * and & 
date: 2025-02-25 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---



## unary * and & 

`*` is for dereferencing and `&` is for getting varible address.

Add two new node type in parser
`ND_ADDR` for `&` and `ND_DEREF` for `*`


For example, when we meet `*x` or `&y` during the parser phase,
we create node type `ND_DEREF` and `ND_ADDR` for each of them.

```c
struct Node* unary(Token** rest, Token* tok) {

    // *
    if(equal(tok, "*")) {
        return new_unary(ND_DEREF, unary(rest, tok->next), tok);
    }

    // &
    if(equal(tok, "&")) {
        return new_unary(ND_ADDR, unary(rest, tok->next), tok);
    }

}
```


During code geneartion phase, for `ND_ADDR` type we call `gen_addr(node->lhs)`,
for `ND_DEREF`, we first call `gen_expr(node->lhs)` to get the value of the `node->lhs`,
and then we generate code `mov (%rax), %rax` which means that it move the value pointed by 
the `%rax` register to `%rax` register.


```c
static void gen_addr(Node* node) {
  switch(node->kind) {
    case ND_VAR:
    printf("  lea %d(%%rbp), %%rax\n", node->obj->offset);
    return;

    case ND_DEREF:
      gen_expr(node->lhs);
      return;
  } 
  error_tok(node->tok, "not an lvalue");
}

static void gen_expr(Node* node) {
    switch (node->kind) {
        // 
        //  *** 
        case ND_ADDR:
        gen_addr(node->lhs);
        return;

        case ND_DEREF:
        gen_expr(node->lhs);
        print(" mov (%%rax), %%rax\n");
        return ;


    }
}


```

