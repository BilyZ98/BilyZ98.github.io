---
layout: post
title: cpp thread local   
date: 2024-08-05 07:59:00-0400
description:  
tags:   cpp  
categories: cpp
featured: false
---



[thread-local](https://github.com/muluoleiguo/interview/blob/master/%E9%9D%A2%E8%AF%95/c%2B%2B%E5%B9%B6%E5%8F%91/%E5%A4%9A%E7%BA%BF%E7%A8%8B/%E7%BA%BF%E7%A8%8B%E6%9C%AC%E5%9C%B0%E5%AD%98%E5%82%A8.md)


When to use thread local?

When you want to store data that is unique to each thread, you can use thread local storage. This is useful when you want to store data that is global to a thread, but not global to the entire program. For example, you might want to store a counter that is unique to each thread, 
or a pointer to a resource that is unique to each thread.

Usually each thread uses thread local when there are multiple function calls in each thread and each function call needs to access the same data.

```cpp
#include <iostream>
#include <thread>

void add(int n) {
    thread_local int count = 0;
    // static thread_local int count = 0; // 两种写法等价！
    count += n;
    // 休眠n秒，防止输出时数据交错（Mac会出现）
    std::this_thread::sleep_for(std::chrono::seconds(n));
    std::cout<<std::this_thread::get_id()<<":"<<count<<std::endl;
}

int main() {
    std::thread td[2];
    for (int i = 0; i < 2; i++) {
        td[i] = std::thread(add, i+1);
    }
    for (int i = 0; i < 2; i++) {
        td[i].join();
    }
    return 0;
}
```

Output:
```
23456242050624:1
23456239949376:2
```


```cpp
#include <iostream>
#include <thread>

class A {
public:
    void dump() {
        std::cout<<id<<":"<<count<<std::endl;
    }
    std::thread::id id;
    static thread_local int count;
};
thread_local int A::count = 0;

void add(int n) {
    A a;
    a.id = std::this_thread::get_id();
    a.count += n;
    std::this_thread::sleep_for(std::chrono::seconds(n));
    a.dump();
    A aa;
    aa.dump(); // aa 和 a 中的count在同一个线程内相同。
}
int main()
{
    std::thread td[2];
    for (int i = 0; i < 2; i++) {
        td[i] = std::thread(add, i+1);
    }
    for (int i = 0; i < 2; i++) {
        td[i].join();
    }
    return 0;
}
```

Output:
```
23456242050624:1
thread::id of a non-executing thread:1
23456239949376:2
thread::id of a non-executing thread:2
```
