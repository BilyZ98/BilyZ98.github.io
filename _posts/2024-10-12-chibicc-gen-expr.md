---
layout: post
title: Simple c compiler gen expr     
date: 2024-10-09 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---


Once we have the tokenize we can then write parser code 
to generate abstract syntax tree by traversing the token list.
We use stack to generate tree.


`'+'` has the lowest priority.
Number and `(expr)`has highest parsing priority.


```cpp
// expr = mul ("+" mul | "-" mul)*
static Node* expr(Token **rest, Token* tok){
  Node* node= mul(&tok, tok);

  for(;;){
    if(equal(tok, "+") ) {
      node = new_binary(ND_ADD, node, mul(&tok, tok->next));
      continue;
    }

    if(equal(tok, "-")) {
      node = new_binary(ND_SUB, node, mul(&tok, tok->next));
      continue;
    }

    *rest = tok;
    return node;

  }
}

// expr = primary ("*" primary | "/" primary)*
static Node* mul(Token**rest, Token* tok) {
  Node* node = primary(&tok, tok);

  for(;;) {
    if(equal(tok, "*")) {
      node =  new_binary(ND_MUL, node, primary(&tok, tok->next));
      continue;;
    }

    if(equal(tok, "/")) {
      node = new_binary(ND_DIV, node, primary(&tok, tok->next));
      continue;;
    }

    *rest = tok;
    return node;
  }
}

// primary = "(" expr ")" | num
static Node* primary(Token **rest, Token* tok) {
  if(equal(tok, "(")) {
    Node* node = expr(&tok, tok->next);
    *rest = skip(tok, ")");
    return node;
  }

  if(tok->kind  == TK_NUM) {
    Node* node = new_num(tok->val);
    *rest = tok->next;
    return node;
  }

  error_tok(tok, "expected an expression");

}
```

## order of parser is important   
The order of operations in the code snippet you provided is crucial for correctly evaluating the expression represented by the abstract syntax tree (AST). Let's go through the code step-by-step:

```c
gen_expr(node->rhs);
push();
gen_expr(node->lhs);
pop("%rdi");
```

1. **Evaluate the Right Subtree**: `gen_expr(node->rhs);`
   - This recursively generates code for the right-hand side expression of the current node. The result of this expression is expected to be in the `%rax` register.

2. **Push the Result of the Right Subtree**: `push();`
   - The value in `%rax` (which is the result of the right-hand side expression) is pushed onto the stack. This saves the result so that the `%rax` register can be used for other purposes.

3. **Evaluate the Left Subtree**: `gen_expr(node->lhs);`
   - This recursively generates code for the left-hand side expression of the current node. The result of this expression is also expected to be in the `%rax` register.

4. **Pop the Saved Right Subtree Result**: `pop("%rdi");`
   - The value that was previously pushed onto the stack (the result of the right-hand side expression) is popped into the `%rdi` register.

The reason for this specific order is to ensure that the values of the left and right subtrees are correctly placed in the registers for further operations. By convention, the result of the left subtree is left in `%rax`, and the result of the right subtree is placed in `%rdi`. This is a common calling convention for binary operations where the left operand is in `%rax` and the right operand is in `%rdi`.

This order of operations is necessary because the code generator needs to follow the calling convention and ensure that the values are in the correct registers before performing the operation represented by the current node. The `push` and `pop` operations are used to temporarily save and restore the values to maintain the correct evaluation order and register usage.



## quotient and remainder in division

The **quotient** and **remainder** are the results of a division operation. The quotient is the number of times the divisor fits into the dividend, and the remainder is what's left over after the division.

Here's a simple example to illustrate:

Let's say we want to divide 17 by 5:
- **Dividend**: 17
- **Divisor**: 5

When we divide 17 by 5, we get:
- **Quotient**: 3 (because 5 fits into 17 three times)
- **Remainder**: 2 (because after taking away 15, which is 5 times 3, from 17, we have 2 left over)

So, in mathematical terms:
\[ 17 \div 5 = 3 \text{ remainder } 2 \]

In the context of the `idiv` instruction in assembly language:
- The quotient is stored in the `%rax` register.
- The remainder is stored in the `%rdx` register.

This is how the division operation works at a fundamental level, and it's the same concept used in assembly language for the `idiv` instruction. The `idiv` instruction performs a division and stores the quotient and remainder in the appropriate registers.
