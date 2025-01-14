---
layout: post
title: chibicc - Simple c compiler block {} node  
date: 2025-01-13 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---





[Commit history for block feature](https://github.com/BilyZ98/chibicc/commit/50d55515fe3a882f90fec3fbee8b5795239b60f8)

## expected an expression error after adding block {} node

Problem:
```bash
{ a=3; return a; }
   ^ expected an expression
make: *** [Makefile:12: test] Error 1
```


Root cause:
```cpp
// compound_stmt = stmt* "}"
static Node* compound_stmt(Token** rest, Token* tok) {
  Node head = {};
  Node* cur = &head;
  while(!equal(tok, "}")) {
    // This is the bug, should be tok not tok->next;
    cur = cur->next = stmt(&tok, tok->next);
  }
  Node* node = new_node(ND_BLOCK);
  node->body = head.next;
  *rest = tok->next;

  return node;

}
```

Fix:
```cpp
// compound_stmt = stmt* "}"
static Node* compound_stmt(Token** rest, Token* tok) {
  Node head = {};
  Node* cur = &head;
  while(!equal(tok, "}")) {
    cur = cur->next = stmt(&tok, tok);
  }
  Node* node = new_node(ND_BLOCK);
  node->body = head.next;
  *rest = tok->next;

  return node;

}
```

## What is done to introduce block concept?  

Introduce another node in node struct called `body` to store 
the  code content inside block

No change is made to tokenizer

For parser, introduce `compound_stmt` production/grammar rule for
 `stmt` generation rule.


For code generator, start generating code from `Function->body` part which
is a block itself. Each `body` inside each block has its own list of nodes.
Previously we only have on list of nodes. Now we have   one list of nodes for 
each block node.


I guess this is for variable scope purpose, althought this has not been done in this 
commit history.


