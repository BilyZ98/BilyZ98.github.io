---
layout: post
title: Linux get cpu time and wall clock time
date: 2024-10-13 07:59:00-0400
description:  
tags:  c  linux 
categories: os 
featured: false
---




## Difference between wall clock time and cpu time 
The difference between CPU time and system clock time (also known as wall-clock time) is an important concept in performance measurement:

**CPU Time:**
- **User CPU Time**: The amount of time the CPU spends executing code in user mode, outside the kernel.
- **System CPU Time**: The amount of time the CPU spends executing code in kernel mode, on behalf of the process.
- **Total CPU Time**: The sum of user and system CPU time, representing the total time the CPU was actively working on a specific process.

**System Clock Time (Wall-Clock Time):**
- This is the real-world elapsed time from the start to the end of a process, as if measured by a stopwatch.
- It includes all time, including time spent waiting for I/O operations, network responses, or other processes to complete.

**Key Differences:**
- **CPU time** only accounts for the time the CPU is actively working on a task, while **wall-clock time** includes all waiting periods.
- **Wall-clock time** is always greater than or equal to **CPU time** for a single-threaded process.
- For multi-threaded processes, **CPU time** can exceed **wall-clock time** if multiple threads are running in parallel on multiple CPU cores.

**Example:**
If a program takes 2 seconds of CPU time but 5 seconds of wall-clock time, it means the program was waiting for 3 seconds (e.g., for I/O operations or network responses).

Understanding these differences can help identify performance bottlenecks. If CPU time is close to wall-clock time, the process is CPU-bound. If wall-clock time is significantly higher, the process is likely I/O-bound or waiting on other resources.


## Get cpu time in rocksdb
https://github.com/BilyZ98/rocksdb_kv_sep/blob/8a5f06aef1d74d4dace2ffdcd2f07b90ddcff083/db/flush_job.cc#L697

Wall clock time:
`CLOCK_MONOTONIC` is used to get wall clock time

```cpp
 uint64_t NowNanos() override {
#if defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_GNU_KFREEBSD) || \
    defined(OS_AIX)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#elif defined(OS_SOLARIS)
    return gethrtime();
#elif defined(__MACH__)
    clock_serv_t cclock;
    mach_timespec_t ts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &ts);
    mach_port_deallocate(mach_task_self(), cclock);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
  }
```

CPU time:
`CLOCK_THREAD_CPUTIME_ID` is used to get cpu time

https://github.com/BilyZ98/rocksdb_kv_sep/blob/8a5f06aef1d74d4dace2ffdcd2f07b90ddcff083/env/env_posix.cc#L164
```cpp
  uint64_t CPUMicros() override {
#if defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_GNU_KFREEBSD) || \
    defined(OS_AIX) || (defined(__MACH__) && defined(__MAC_10_12))
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec) / 1000;
#endif
    return 0;
  }
```
