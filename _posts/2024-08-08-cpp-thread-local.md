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

```
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


```
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
