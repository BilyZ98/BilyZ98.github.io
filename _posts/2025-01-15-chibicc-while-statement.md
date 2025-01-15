---
layout: post
title: chibicc - Simple c compiler for statement  
date: 2025-01-14 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---


[Commit history of for statement feature](https://github.com/BilyZ98/chibicc/commit/ed1f13abd63cc10e7cbd76c6f6de784df0f801c1)

## What is changed to introduce for loop ?

No big changes on top of if statement feature.

For parser, add another grammar/production rule for `for` statement.
```cpp
// stmt = "return" expr ";" 
//        | "{" compound_stmt
//        | expr_stmt
//        | "if" "(" expr ")" stmt ("else" stmt)?
//        | "for" "(" expr_stmt expr? ";" expr? ")" stmt 
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

  if(equal(tok, "for")) {
    tok = skip(tok->next, "(");
    Node* node = new_node(ND_FOR);
    node->init = expr_stmt(&tok, tok);
    if(!equal(tok, ";")) {
      node->cond = expr(&tok, tok);
    }
    tok = skip(tok, ";");

    if(!equal(tok, ")")) {
    node->inc = expr(&tok, tok);
    }
    tok = skip(tok, ")");
    node->then = stmt(&tok, tok);
    *rest = tok;
    return node;
  }

  return expr_stmt(rest, tok);
}

```

Introduce `init` and `inc` node inside of `Node` type to represent initialization
and increment operation in `for` statement.
`for(init;cond; inc){}` 
```c
typedef struct Node Node;
struct Node {
  NodeKind kind;
  Node* lhs; //left hand side
  Node* rhs; // right hand side
  Node* next; // next node
  Node* body; // {} body node

  // "if" or "for" statement
  Node* cond; 
  Node* then; 
  Node* els; 
  Node* init;
  Node* inc;


  // char name;
  Obj* obj; // used if kind == ND_VAR
  int val;  // used if kind == ND_NUM
};

```



For code generator, generate `.L.begin.%d:` to indicate the start of the for block.
Use `cmp $0, %%rax` and `jmp .L.end.%d` after `cond` to go out of for block.
Use `jmp .L.begin.%d` to jmp back to the begining of the for block at the end. 
And then comparison at the begining will decide whether to jump out of the 
for block or not.

```c
  case ND_FOR: {
      int c = count_depth();
      gen_stmt(node->init);
      printf(".L.begin.%d:\n", c);
      if(node->cond) {
        gen_expr(node->cond);
        printf("  cmp $0, %%rax\n");
        printf("  je .L.end.%d\n",c);
      }

      gen_stmt(node->then);
      if(node->inc) {
        gen_expr(node->inc);
      }
      printf("  jmp .L.begin.%d\n",c);
      printf(".L.end.%d:\n",c);
      return;
    }
```
