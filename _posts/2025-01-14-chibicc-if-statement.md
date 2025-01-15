---
layout: post
title: chibicc - Simple c compiler if statement  
date: 2025-01-14 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---


[Commit history for if statement feature](https://github.com/BilyZ98/chibicc/commit/2f132cf8e68f0adf92bae038b75ea6da425e223c)

## What is changed ?

For parser, new node type called `ND_IF` is introduced.
Three new nodes are introduced for `Node` type in parser.
They are called `cond`, `then`, `els` which corresponds to 
code in `if(cond){ } else {}`.
New production rule is introduced to deal with `if` statement
```cpp
// stmt = "return" expr ";" 
//        | "{" compound_stmt
//        | expr_stmt
//        | "if" "(" expr ")" stmt ("else" stmt)?
static Node* stmt(Token**rest, Token* tok) {
  if(equal(tok, "return")) {
    Node* node = new_unary(ND_RETURN, expr(&tok, tok->next));
    *rest = skip(tok, ";");
    return node;
  }

  if(equal(tok, "{")) {
    return compound_stmt(rest, tok->next);
  }

  if(equal(tok, "if")) {
    tok = skip(tok->next, "(");
    Node* node = new_node(ND_IF);
    node->cond = expr(&tok, tok);
    tok = skip(tok, ")");
    node->then = stmt(&tok, tok);
    if(equal(tok, "else")) {
      node->els = stmt(&tok, tok->next);
    }
    *rest = tok;
    return node;
  }

  return expr_stmt(rest, tok);
}
```


For code generator, assembly code generation for `if` condition is introduced in
`gen_stmt`. First we generate assembly code for `cond` node, and then we generate `je $0, %rax` 
to check condtion of `cond` node, and call `jump .L.else.%d` to do them instruction jump,
`%d` is used to uniquly identify each `else` block. Because multiple `if` statement can be nested 
at the same time.


