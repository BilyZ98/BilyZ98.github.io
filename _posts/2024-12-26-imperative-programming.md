---
layout: post
title: Imperative programming vs. Declarative programming 
date: 2024-12-24 07:59:00-0400
description:  
tags:  programming    
categories: interview 
featured: false
---



## What is imperative programming vs. declarative programming

Imperative programming means that each steps of command/code is specified to 
get the final computation result.

Declarative programming means that only the results that are wanted are specified,
no step by step code is provided to get the final computation results.


For example,

SQL is a declarative programming language

```sql
select * from user 
where name = 'bily'
```


c++ is an iimperative programming language
```cpp
vector<string> users;
    for(int i=0; i < users.size(); i++) {
        if(users[i] == "bily") {
            cout << user[i] << endl;
        }
}
```


### Advantage and disadvantage of imperative programming.
1. Imperative programming depends on defined instructions to achieve final 
computation results, so code can be easy to understand and straightforward.

2. Order of performed operations is completely controlled by developer 

3. Bugs can be easily traced because the program is assembled from blocks of code
that is based on step by step commands.

4. Memory allocation and manipulation is directly linked in imperative programming.
Efficient use of machine memory.

### Reference
[1](https://zhuanlan.zhihu.com/p/34445114)
[2](https://www.geeksforgeeks.org/what-is-imperative-programming/)

