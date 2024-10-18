---
layout: post
title: C compiler - single letter local variable
date: 2024-10-15 07:59:00-0400
description:  
tags:  c compiler 
categories: compiler
featured: false
---



## Add single letter identity in c compiler
Need to allocate memory address for single letter variable.

Use following two registers.
`rsp`
`rbp`

In x86 assembly language, `rsp` and `rbp` are registers that play a crucial role in stack management and function calls:

- **`rsp` (Stack Pointer)**: This register points to the top of the stack, which is a region of memory used for dynamic storage during program execution. The stack pointer is adjusted automatically when values are pushed onto or popped off the stack. It's typically used to keep track of the current position in the stack.

- **`rbp` (Base Pointer)**: This register is used as a reference point for the stack frame of a function. A stack frame contains the function's local variables and saved registers. The base pointer is usually set to the value of the stack pointer at the beginning of a function and remains constant throughout the function's execution. This allows for easy access to function parameters and local variables via fixed offsets from `rbp`.

Here's a brief example to illustrate their usage in a function call:

```assembly
push rbp            ; Save the caller's base pointer
mov rbp, rsp        ; Set the base pointer to the current stack pointer
sub rsp, 16         ; Allocate space for local variables
; Function body
mov rsp, rbp        ; Restore the stack pointer
pop rbp             ; Restore the caller's base pointer
ret                 ; Return to the caller
```

In this example:
- The caller's base pointer is saved on the stack.
- The base pointer is set to the current stack pointer, establishing a new stack frame.
- Space is allocated on the stack for local variables.
- At the end of the function, the stack pointer and base pointer are restored, and control is returned to the caller.

These registers are essential for implementing nested function calls and maintaining a clean and organized stack structure. If you're interested in learning more about x86 assembly and stack management, I can provide further details or examples. Just let me know! 😊


`mov (%%rax), %%rax` moves value in `%rax` address memory to %rax

`lea 8(%%rbp), %%rax` calculate the effective address `8+%rbp` and then load 
effective address (not memory content) into `%%rax` 


### Parse
Add assign expression and equality expression.
```cpp
// expr = assign
static Node* expr(Token **rest, Token* tok){
  return assign(rest, tok);
}

// assign = equality ("=" assign)?
static Node* assign(Token** rest, Token* tok) {
  Node* node = equality(&tok, tok);
  if(equal(tok, "=")) {
    node = new_binary(ND_ASSIGN, node, assign(&tok, tok->next));
  }

  *rest = tok;
  return node;
}
```


Add ident type in primary expression.
```cpp
// primary = "(" expr ")" | ident | num
static Node* primary(Token **rest, Token* tok) {
  if(equal(tok, "(")) {
    Node* node = expr(&tok, tok->next);
    *rest = skip(tok, ")");
    return node;
  }

  if(tok->kind == TK_IDENT) {
    Node* node = new_var_node(*(tok->loc));
    *rest = tok->next;
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

### Codegen
Add two new node types.
1. `ND_VAR` for loading memory address of variable.
2. `ND_ASSIGN` for get the value of right hand side expression and 
assign the value to memory address of the left hand side variable.

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

    case ND_VAR:
    gen_addr(node);
    printf("  mov (%%rax), %%rax\n");
    return;;

    case ND_ASSIGN:
    gen_addr(node->lhs);
    push();
    gen_expr(node->rhs);
    pop("%rdi");
    printf("  mov %%rax, (%%rdi)\n");
    return;
  }

```

Only use stack memory to store value of variable.
Each single letter variable takes 8 bytes memory space.
```cpp
static void gen_addr(Node* node) {
  if(node->kind == ND_VAR) {
    int offset = (node->name - 'a' + 1)*8;
    printf("  lea %d(%%rbp), %%rax\n", -offset);
    return;
  }

  error("not an lvalue");
}
```
