---
layout: post
title: Computer basics  
date: 2024-12-21 07:59:00-0400
description:  
tags:  algorithm os linux cpp lock b+tree
categories: interview 
featured: false
---



Had an interview today.

I was struggling to answer the computer basics questions smoothly.

I write down the questions I got here to help strengthen the knowledge.

## Difference between b+ tree, b tree and binary tree

- b+ tree and btree can be used to disk storage, binary tree is for memory opeartion.
### Difference between b+tree and btree
- Structure and storage
    btree stores both keys and data pointers in leaf nodes and internal nodes.

    b+tree only stores data pointers at leaf nodes. Internal nodes are only used to store keys go guide search
    process

- Search efficiency
    Search in btree can be slower because data might be in any level in btree

    Search in b+tree might be faster because all data is stored in leaf nodes

- Insertion and deletion
    Insertion and deletion in b=tree is more simpler because they only affect leaf nodes.

- Sequential access
    Sequential access is not possible in a btree because leaf nodes are not linked.

    It's possible in b+tree

- Applications
    btree used in databases and search engines

    b+tree used in database indexing and multi-level indexing

### Where is binary tree used in real worlod software
Linux kernel.

1. I/O scheduler
    CFQ(completely fair queueing) I/O scheduler to distribute 
    I/O bandwidth among all I/O requests.
2. Filesystem

3. virtual memory area

4. epoll file descriptor

5. network packets

## mutex and spinlock
mutex.lock() will put thread to sleep if it can not get the lock immediately 
which is blocking lock acquire

spinlock.lock() will not put thread into sleep. It does busy checking all the time 
until it gets the lock.

mutex internal implementation with pthread

As we can see it uses conditional variable to do this sleep and wakeup.
```cpp
int pthread_mutex_lock(pthread_mutex_t *mutex) {
    // Try to acquire the lock using an atomic operation
    if (atomic_compare_and_swap(&mutex->lock, 0, 1) == 0) {
        // Successfully acquired the lock
        return 0;
    }

    // Spin for a short period
    while (spin_count < MAX_SPIN_COUNT) {
        if (atomic_compare_and_swap(&mutex->lock, 0, 1) == 0) {
            // Successfully acquired the lock
            return 0;
        }
        spin_count++;
    }

    // If spinning fails, block the thread
    block_thread(mutex);

    // When the thread is woken up, try to acquire the lock again
    while (atomic_compare_and_swap(&mutex->lock, 0, 1) != 0) {
        // Wait until the lock is available
        wait_for_lock(mutex);
    }

    return 0;
}


#include <pthread.h>
#include <queue>
#include <atomic>

struct Mutex {
    std::atomic<bool> locked;
    std::queue<pthread_t> waiting_threads;
    pthread_mutex_t internal_mutex;
    pthread_cond_t cond;
};

void block_thread(Mutex* mutex) {
    pthread_mutex_lock(&mutex->internal_mutex);

    // Add the current thread to the waiting queue
    pthread_t current_thread = pthread_self();
    mutex->waiting_threads.push(current_thread);

    // Wait for the condition variable to be signaled
    while (mutex->locked.load()) {
        pthread_cond_wait(&mutex->cond, &mutex->internal_mutex);
    }

    // Remove the current thread from the waiting queue
    mutex->waiting_threads.pop();

    pthread_mutex_unlock(&mutex->internal_mutex);
}

void wake_thread(Mutex* mutex) {
    pthread_mutex_lock(&mutex->internal_mutex);

    if (!mutex->waiting_threads.empty()) {
        // Signal the condition variable to wake up one waiting thread
        pthread_cond_signal(&mutex->cond);
    }

    pthread_mutex_unlock(&mutex->internal_mutex);
}

```



