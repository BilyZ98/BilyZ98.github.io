# Lock

## How to do distributed locking 
[How to do distributed locking](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html)
Do not use redlock which uses lease on top of redis.
Because this makes assumption about underlying clock system.
To make sure correct use of distributed lock, please 
use zookeeper or other concensus tool to implement strict clock increasing.
And use strictly increasing fencing token for data server to decide whether 
to accept write or not.

##  shared_lock 
What is shared_lock? 
shared_lock can allow concurrent read but only allows one
writer at a time.

Implementation

```cpp
#include <atomic>

class my_shared_mutex {
    std::atomic<int> refcount {0};
public:
    void lock() // write lock
    {
        int val;
        do {
            val = 0; // Can only take a write lock when refcount == 0
        } while (!refcount.compare_exchange_weak(val, -1, std::memory_order_acquire));
    }

    void unlock() // write unlock
    {
        refcount.store(0, std::memory_order_release);
    }

    void lock_shared() // read lock
    {
        int val;
        do {
            do {
                val = refcount.load(std::memory_order_relaxed);
            } while (val == -1); // spinning until the write lock is released
        } while (!refcount.compare_exchange_weak(val, val+1, std::memory_order_acquire));
    }

    void unlock_shared() // read unlock
    {
        refcount.fetch_sub(1, std::memory_order_relaxed);
    }
};

```

## One function call no matter how many instances calling 
```
class MyClass {
public:
    int myFunc() {
        static bool myFuncHasBeenCalled = false;
        if (myFuncHasBeenCalled) {
            return 0;
        } else {
            myFuncHasBeenCalled = true;
            return 1;
        }
    }
};

```

Thread-safe implementation
The initialization of local static variables in C++ is thread-safe since C++11123. This means that the constructor of the static variable is guaranteed to run only once, even in a multithreaded environment1. If control enters the declaration concurrently while the variable is being initialized, the concurrent execution shall wait for completion of the initialization1.

However, it’s important to note that subsequent access to the variable is not guaranteed to be thread-safe1. If multiple threads are accessing and modifying the static variable, you may need to use synchronization mechanisms like mutexes to ensure thread safety.
```
#include <mutex>

class MyClass {
public:
    int myFunc() {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);

        static bool myFuncHasBeenCalled = false;
        if (myFuncHasBeenCalled) {
            return 0;
        } else {
            myFuncHasBeenCalled = true;
            return 1;
        }
    }
};

```
## Mutex implementation 

```
#include <atomic>
#include <condition_variable>
#include <thread>

class Mutex {
private:
    std::atomic<bool> locked{false};
    std::condition_variable cv;
    std::mutex cv_m;

public:
    void lock() {
        bool expected = false;
        // Try to acquire the lock
        while (!locked.compare_exchange_strong(expected, true)) {
            expected = false;
            std::unique_lock<std::mutex> lk(cv_m);
            // Wait for the lock to be released
            cv.wait(lk, [&] { return !locked.load(); });
        }
    }

    void unlock() {
        // Release the lock
        locked.store(false);
        cv.notify_one();
    }
};

void shared_function(Mutex& m) {
    m.lock();
    // Critical section
    m.unlock();
}

int main() {
    Mutex m;
    std::thread t1(shared_function, std::ref(m));
    std::thread t2(shared_function, std::ref(m));

    t1.join();
    t2.join();

    return 0;
}

```

## Spinlock implementation 

; Intel syntax
locked: ; The lock variable. 1 = locked, 0 = unlocked.
dd 0

spin_lock:
mov eax, 1 ; Set the EAX register to 1.
xchg eax, [locked] ; Atomically swap the EAX register with the lock variable.

