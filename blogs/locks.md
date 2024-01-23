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

