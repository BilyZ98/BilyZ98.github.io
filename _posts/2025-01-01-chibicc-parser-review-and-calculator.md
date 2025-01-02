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


## Simple math expression evaluator

Problem:

Implement math expresion evaluator for expression containing `+,-,*,()`

Solution:

This problem is very typical and important. It is a problem that uses stack 
to solve the priority issue between different operator like `+` and `*`.

Since `*` has higher priority than `+`, so when we meet `+` during string scan 
we can not  evalute it immediately because there might be a `*` following this `+`.

For example `3+2*3`, we should evaluate `2*3` first.

[ref two stack implementation for expression evaluator](https://www.nowcoder.com/practice/c215ba61c8b1443b996351df929dc4d4)

1. for `(` , we just push it into ops stack

2. for ')' , we keep evaluating expression until `(` is met on `ops` stack

3. for numbers, we just push it into nums stack.

4. for `+,-,*`, we evaluate the expression in `ops` stack if priority of 
the op of the top of `ops` stack is equal or higher than current op in string.

Pay attention that we need to stop if top of the `ops` stack is `(`, because 
`()` has highest priority.

Code: 
```cpp


#include <cctype>
#include <string>
#include <iostream>
#include <stack>
#include <unordered_map>
using namespace std;

unordered_map<char, int> op_pri = {
  {'+',1},
  {'-', 1},
  {'*',2}
};

int cal_one(stack<char>& ops, stack<int>& nums) {
  int b = nums.top(); 
  nums.pop();
  int a = nums.top();
  nums.pop();

  char c  = ops.top();
  ops.pop();
  int num = 0;
  if(c == '+') {
    num = a + b;

  } else if(c == '-') {
    num = a - b;
  } else if(c == '*') {
    num = a * b;
  }

  return num;
}
int solve(std::string s) {
  stack<char> ops;
  stack<int> nums;

  nums.push(0);
  for(int i=0; i < s.length(); i++) {
    char c= s[i];

    if(c == '(') {
      ops.push(c);
    } else if(c == ')') {
      while(1) {
        if(ops.top() != '(') {
          int res = cal_one(ops, nums);
          nums.push(res);
        } else {
          ops.pop();
          break;
        }
      }
    } else {
      // digit
      if(isdigit(c)) {
        int j = i;
        int cur_num = 0;
        while(j < s.length() && isdigit(s[j])) {
          cur_num = cur_num * 10 + s[j]-'0';
          j++;
        }
        i = j-1;
        nums.push(cur_num);
      } else {

        // + - *
        if(i > 0 && (s[i-1] == '(' || s[i-1] == '-' || s[i-1]=='+')) {
          nums.push(0);
        }
        // This ops.top() != '(' is important. (expr) is highest priority
        while(!ops.empty() && ops.top() != '(') {
          if(op_pri[ops.top()] >= op_pri[c]  ) {
            int res = cal_one(ops, nums);
            nums.push(res);
          } else {
            break;
          }
        }
        ops.push(c);
      }

    }
  }
  while(!ops.empty()) {
    int res = cal_one(ops, nums);
    nums.push(res);
  }

  return nums.top();
  return 0;

}
int main(int argc, char* argv[]) {


  string s(argv[1]);


  cout << s << endl;

  cout << solve(s) << endl;

  return 0;


}
```

Test case:
```bash
(base) ➜  chibicc git:(main) ✗ g++ math_eval.cc
(base) ➜  chibicc git:(main) ✗ ./a.out "1"
1
1
(base) ➜  chibicc git:(main) ✗ ./a.out "2"
2
2
(base) ➜  chibicc git:(main) ✗ ./a.out "2+1"
2+1
3
(base) ➜  chibicc git:(main) ✗ ./a.out "2+2*5"
2+2*5
12
(base) ➜  chibicc git:(main) ✗ ./a.out "2+2*5-8"
2+2*5-8
4
(base) ➜  chibicc git:(main) ✗ ./a.out "2+2*(5-8)"
2+2*(5-8)
-4
(base) ➜  chibicc git:(main) ✗ ./a.out "2+2*(5-8)*3"
2+2*(5-8)*3
-16
(base) ➜  chibicc git:(main) ✗ ./a.out "2+2*(5-8*2)"
2+2*(5-8*2)
-20
(base) ➜  chibicc git:(main) ✗ ./a.out "2+2*(5*8-2)"
2+2*(5*8-2)
78
```
