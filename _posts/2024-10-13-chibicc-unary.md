---
layout: post
title: Simple c compiler unary      
date: 2024-10-10 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---


## Add unary to parser
[Previous parser post](./2024-10-12-chibicc-gen-expr.md) 

Add unary symbol in mul. Add unary node kind and replace primary
with unary. 
unary is a super set of unary and primary.

```cpp
// expr = unary ("*" unary | "/" unary)*
static Node* mul(Token**rest, Token* tok) {
  Node* node = unary(&tok, tok);

  for(;;) {
    if(equal(tok, "*")) {
      node =  new_binary(ND_MUL, node, unary(&tok, tok->next));
      continue;;
    }

    if(equal(tok, "/")) {
      node = new_binary(ND_DIV, node, unary(&tok, tok->next));
      continue;;
    }

    *rest = tok;
    return node;
  }
}


// unary = ("+" | "-") unary | primary
static Node* unary(Token **rest, Token* tok) {
  if(equal(tok, "+"))
    return unary(rest, tok->next);

  if(equal(tok, "-"))
    return new_unary(ND_NEG, unary(rest, tok->next));

  return primary(rest, tok);

}
```

