---
layout: post
title: chibicc C compiler - multi char variable name
date: 2025-01-05 07:59:00-0400
description:  
tags:  c compiler cpp   
categories: compiler
featured: false
---




### Find bug in code that leads to seg fault.

[https://github.com/BilyZ98/chibicc/commit/3ca91cc6431246e1b23f4503b6442e77e7457246](https://github.com/BilyZ98/chibicc/commit/3ca91cc6431246e1b23f4503b6442e77e7457246)
The bug code is at parse.c after I use gdb
```
77      ../sysdeps/x86_64/multiarch/strlen-evex.S: No such file or directory.
(gdb) bt
#0  __strlen_evex () at ../sysdeps/x86_64/multiarch/strlen-evex.S:77
#1  0x0000555555555ff7 in find_var (start=0x7fffffffe399 "a;", len=1) at parse.c:194
#2  0x0000555555556140 in primary (rest=0x7fffffffdd30, tok=0x55555555a360) at parse.c:220
#3  0x0000555555555fbf in unary (rest=0x7fffffffdd30, tok=0x55555555a360) at parse.c:187
#4  0x0000555555555e73 in mul (rest=0x7fffffffdd60, tok=0x55555555a360) at parse.c:160
#5  0x0000555555555d9a in add (rest=0x7fffffffdd90, tok=0x55555555a360) at parse.c:138
#6  0x0000555555555c1f in relational (rest=0x7fffffffddc0, tok=0x55555555a360) at parse.c:109
#7  0x0000555555555b46 in equality (rest=0x7fffffffddf0, tok=0x55555555a360) at parse.c:90
#8  0x0000555555555ac1 in assign (rest=0x7fffffffde40, tok=0x55555555a360) at parse.c:78
#9  0x0000555555555a98 in expr (rest=0x7fffffffde40, tok=0x55555555a360) at parse.c:73
#10 0x0000555555555a3d in expr_stmt (rest=0x7fffffffde98, tok=0x55555555a360) at parse.c:65
#11 0x00005555555559e4 in stmt (rest=0x7fffffffde98, tok=0x55555555a360) at parse.c:46
#12 0x0000555555556248 in parse (tok=0x55555555a360) at parse.c:245
#13 0x00005555555558c9 in main (argc=2, argv=0x7fffffffe038) at main.c:9
```

function
```cpp
static Obj* find_var(char* start, int len) {
  for(Obj* o_idx = local_obj_ptr; o_idx; o_idx=o_idx->next) {
    if(len == strlen(o_idx->name) && strncmp(start, o_idx->name, len) == 0) {
      return o_idx;
    }
  }
  return NULL;
}
```


Finally found the bug.

I did not include this define at `chibicc.h`

lol.
```
#define _POSIX_C_SOURCE 200809L
```

This line of code enable functions like `strndup, getline, and clock_gettime`
which are not part of the standard C library but are available in the POSIX 
standard. fuck that. Spend really long time on this.





