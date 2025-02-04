



[lock free queue impl on wechat subscription post](https://mp.weixin.qq.com/s/-9RJgcpamtXZOFflOkkt5w)


## Why lock free? 
To avoid lock contention between different threads.


## How to implement it ?

### List implementation



Is this part of code necessary for correct lock free queue implementation ?
```cpp
if (next!= nullptr) {
    tail.compare_exchange_weak(oldTail, next);
    continue;
}
```


Yes, the following part of the code is **necessary** in a **lock-free queue implementation** to maintain correctness and avoid a stuck state:  

```cpp
if (next!= nullptr) {
    tail.compare_exchange_weak(oldTail, next);
    continue;
}
```

### **Why is this necessary?**
This block helps in **correctly advancing the `tail` pointer** when it **lags behind** the actual last node due to contention. Without it, the queue could become stuck in an inefficient state where the `tail` pointer does not point to the actual last node, which may cause issues in subsequent enqueue operations.

---

### **Example of Why This Is Needed**  
Assume a **multi-threaded** environment where multiple threads are enqueuing elements.

#### **Step-by-Step Scenario**
1. **Initial Queue State**  
   ```
   Head → A (tail) → nullptr
   ```
   - `tail` points to **A**.
   - `A->next` is `nullptr`.

2. **Thread T1 begins enqueue(B)**
   - It **gets `oldTail = A`**.
   - It **gets `next = nullptr`**, so it proceeds to insert `B`.

3. **Thread T1 is paused (context switch).**
   - `oldTail->next.compare_exchange_weak(next, newNode)` is **not executed yet**.
   - Queue state is unchanged.

4. **Thread T2 starts enqueue(C)**
   - It also **gets `oldTail = A`**.
   - **Another Thread T3 already completed enqueue(B)**, so now:
     ```
     Head → A → B (tail) → nullptr
     ```
   - Now, `A->next` is `B`, but `tail` is **still pointing to A**.

5. **Thread T2 sees that `next != nullptr` (`next = B`)**
   - It executes:
     ```cpp
     tail.compare_exchange_weak(oldTail, next);
     ```
     - `tail` is updated from **A to B**.

6. **Thread T2 retries the enqueue with `tail = B`**, allowing proper insertion of `C` after `B`.

---

### **What Happens If This Block Is Removed?**
- If `tail.compare_exchange_weak(oldTail, next);` is **not** performed, `tail` would continue pointing to **A**, even though `A->next = B` exists.
- This means **new enqueue attempts might repeatedly try inserting after A instead of after B**.
- In extreme cases, this can lead to a **stalled queue** where elements are not appended correctly.

---

### **Conclusion**
The block:
```cpp
if (next!= nullptr) {
    tail.compare_exchange_weak(oldTail, next);
    continue;
}
```
is necessary to ensure that `tail` **keeps up with the actual last node**, preventing it from getting stuck in an outdated position, especially in a concurrent environment.

Would you like me to clarify anything further? 🚀




My problematic implementation:
```cpp
class MyLockFreeQueue {
public:
  MyLockFreeQueue()  {
    SimpleNode* node = new SimpleNode();
    node->next = nullptr;
    head_ = tail_ = node;

  }

  void enque(int val) {
    std::unique_ptr<SimpleNode> node(new SimpleNode());
    node->val = val;
    SimpleNode* old_tail = nullptr;
    SimpleNode* next  = nullptr;  
    do {
      old_tail = tail_.load();
      next = old_tail->next.load();
      if(old_tail != tail_.load()) {
        continue;
      
      }

      if(next != nullptr) {
        tail_.compare_exchange_weak(old_tail, next);
        continue;
      }

    } while(old_tail->next.compare_exchange_weak(next, node.get()) == false);
    tail_.compare_exchange_weak(old_tail, node.release());
  }

  bool deque(int& val) {
    SimpleNode* old_head = nullptr;
    SimpleNode* old_tail = nullptr;
    SimpleNode* next = nullptr;
    do {
      old_head = head_.load();
      old_tail = tail_.load();

      // assert(old_head != nullptr);
      // This line of code is bug
      next = old_head->next.load();
      if(old_head != head_.load()) {
        continue;
      }
      if(old_head == old_tail) {
        if(next == nullptr) {
          return false;
        }
        tail_.compare_exchange_weak(old_tail, next);
        continue;
      }
      val = next->val;
    } while(head_.compare_exchange_weak(old_head, next) == false);
    delete old_head;
    return true;
  }


  bool empty() {
    return head_.load()->next.load() == nullptr;
  }


  private:
  // size
  std::atomic<int> size_;
  std::atomic<SimpleNode*> head_;
  std::atomic<SimpleNode*> tail_;
};
```

The problem is at deque part.
enque has bug in it.

Fixed code given by gpt 
Your `enque()` function has **several serious bugs** related to **memory management, atomic operations, and ABA problems**. Let's analyze and fix them.

---

## **🔍 Identified Bugs**
### **1. Memory Leak Due to `std::unique_ptr::release()`**
```cpp
std::unique_ptr<SimpleNode> node = std::make_unique<SimpleNode>(val);
```
- `node.release()` transfers ownership but **does not delete the old pointer** if the enqueue operation fails.
- If `compare_exchange_weak()` fails, the `SimpleNode` is **lost** → **memory leak**.

✅ **Fix:** Use `std::shared_ptr` or manually manage deletion.

---

### **2. ABA Problem in `tail_.compare_exchange_weak()`**
```cpp
if(next != nullptr) {
    tail_.compare_exchange_weak(old_tail, next);
    continue;
}
```
- If another thread modifies `tail_` between these operations, it **could still hold the same value**, leading to a **stale pointer reference**.
- This introduces **undefined behavior**.

✅ **Fix:** Use a **hazard pointer** (complex) or ensure `next` is updated **before** modifying `tail_`.

---

### **3. Incorrect Order of `compare_exchange_weak()`**
```cpp
while(old_tail->next.compare_exchange_weak(next, node.get()) == false);
```
- This assumes `next` remains the same between checks, but **other threads could modify it**.
- This can lead to an **infinite loop or inconsistent state**.

✅ **Fix:** Re-load `next` before retrying.

---

### **4. Potential Infinite Loop if `tail_` Changes Midway**
```cpp
if(old_tail != tail_.load()) {
    continue;
}
```
- This check is **not enough** to prevent starvation, as `tail_` may change **again** before re-entering the loop.

✅ **Fix:** Ensure a strong atomic operation.

---

## **✅ Fixed Code**
```cpp
void enque(int val) {
    // Allocate the new node
    std::unique_ptr<SimpleNode> node = std::make_unique<SimpleNode>(val);

    SimpleNode* old_tail = nullptr;
    SimpleNode* next = nullptr;

    while (true) {
        old_tail = tail_.load();
        next = old_tail->next.load();

        if (old_tail != tail_.load()) {
            continue; // Retry if tail changed
        }

        if (next != nullptr) {
            // Help advance tail if it's stale
            tail_.compare_exchange_weak(old_tail, next);
            continue;
        }

        // Try to attach the new node
        if (old_tail->next.compare_exchange_weak(next, node.get())) {
            // Successfully linked, now update tail_
            tail_.compare_exchange_weak(old_tail, node.release());
            return;
        }
    }
}
```

---

## **🔑 Summary of Fixes**
| **Bug** | **Issue** | **Fix** |
|---------|----------|---------|
| **Memory Leak** | `node.release()` without cleanup | Use `std::unique_ptr` safely |
| **ABA Problem** | `tail_.compare_exchange_weak()` fails if another thread modifies `tail_` | Ensure `next` is updated before modifying `tail_` |
| **Order of Operations** | `compare_exchange_weak()` on `next` may lead to an **infinite loop** | Always reload `next` before retrying |
| **Stale `tail_` Pointer** | `if (old_tail != tail_.load())` check is **not strong enough** | Use proper atomic operations |

This new implementation **ensures correctness, prevents memory leaks, and avoids infinite loops**. 🚀


### What is ABA problem ? 
The **ABA problem** in software engineering is a well-known issue in **concurrent programming**, particularly in **lock-free data structures** that rely on **compare-and-swap (CAS) operations**.

### **Understanding the ABA Problem**
1. **CAS (Compare-And-Swap) Basics**  
   - CAS is an atomic operation used in multi-threading to update a variable only if its value hasn't changed since the last check.
   - Example:  
     ```c
     if (atomic_value == expected_value) {
         atomic_value = new_value;
     }
     ```
   - It ensures that updates happen only when no other thread has modified the value.

2. **The Problem**  
   - If one thread (Thread 1) reads a value (e.g., **A**) from a shared variable, another thread (Thread 2) might change it to **B** and then back to **A** before Thread 1 proceeds.
   - CAS sees **A** as unchanged and allows Thread 1 to proceed, even though the value was modified in between.
   - This can lead to **unexpected behavior** and **data inconsistencies**.

### **Example of the ABA Problem**
#### **Scenario**
1. **Thread 1** reads a value **A** from memory.
2. **Thread 2** changes **A → B → A** (some operation happens in between).
3. **Thread 1** now uses CAS to update **A → C**, thinking nothing changed.
4. CAS succeeds because the value is still **A**, but the actual data has been modified!

#### **Why is this Dangerous?**
- If a data structure (like a linked list) relies on CAS for safe updates, the ABA problem may lead to **memory corruption**, **lost updates**, or **dangling pointers**.

### **Solutions to the ABA Problem**
1. **Tagged Pointers (Version Numbers)**  
   - Instead of just storing **A**, store a **version number** along with the value (e.g., `A1`, `B2`, `A3`).
   - When CAS checks, it verifies both the **value** and the **version**.
   - Example:
     ```c
     struct AtomicValue {
         int value;
         int version;
     };
     ```
   - Even if the value is back to **A**, the version number ensures CAS detects the change.

2. **Hazard Pointers**  
   - Used in **memory reclamation** to track objects that are still being accessed by threads, preventing premature deletion.

3. **Read-Copy-Update (RCU)**  
   - Instead of modifying shared data directly, RCU allows **readers to access old versions** while writers update a new copy.

4. **Double Compare-And-Swap (Double CAS or DCAS)**  
   - Some architectures support **checking two variables at once** (e.g., the value + a version counter).

### **Where is the ABA Problem Common?**
- **Lock-free stacks, queues, and linked lists**
- **Garbage collection mechanisms**
- **Reference counting**
- **Memory allocators**


## Why continue is necessary in the second if statement ?

```cpp
  void enque(int val) {
    std::unique_ptr<SimpleNode> node(new SimpleNode());
    node->val = val;
    SimpleNode* old_tail = nullptr;
    SimpleNode* next  = nullptr;  
    while(true) {
      old_tail = tail_.load();
      next = old_tail->next.load();
      if(old_tail != tail_.load()) {
        continue;
      
      }
      if(next != nullptr) {
        tail_.compare_exchange_weak(old_tail, next);
        continue;
      }

      if(old_tail->next.compare_exchange_weak(next, node.get())) {

      tail_.compare_exchange_weak(old_tail, node.release());
        return;
      }
    } 
  }
```
The **`continue`** statement in the second `if` condition is necessary to ensure that the `while(true)` loop properly retries the enqueue operation when the `tail_` pointer is updated asynchronously by another thread.

### **Breaking Down the Code**
This function implements a **lock-free queue enqueue operation** using **atomic operations** (`compare_exchange_weak`) to avoid locks.

#### **Key Elements in the Code:**
- `tail_`: A **lock-free atomic pointer** to the last node in the queue.
- `next`: The next node after `old_tail`, used to check if `tail_` is up-to-date.
- `compare_exchange_weak`: Used for **atomic updates**, but it may fail if another thread modifies the value simultaneously.

### **Understanding the ABA Problem and the Need for `continue`**
1. **First `if`: Detects Stale `tail_`**
   ```cpp
   if(old_tail != tail_.load()) {
       continue;
   }
   ```
   - This check ensures that `old_tail` is still the actual tail.
   - If another thread updates `tail_` during execution, the loop **retries** to reload the new tail.

2. **Second `if`: If `next != nullptr`, Another Thread is Advancing `tail_`**
   ```cpp
   if(next != nullptr) {
       tail_.compare_exchange_weak(old_tail, next);
       continue;
   }
   ```
   - This means **another thread is in the middle of an enqueue operation** but hasn’t yet updated `tail_` completely.
   - The queue is **inconsistent** at this point:  
     - `old_tail->next` points to `next`, but `tail_` still points to `old_tail`.
     - Our thread helps move `tail_` forward to `next` before retrying.
   - The `continue` ensures that we **restart** the loop with a fresh state.

3. **Final Atomic Insert Attempt**
   ```cpp
   if(old_tail->next.compare_exchange_weak(next, node.get())) {
       tail_.compare_exchange_weak(old_tail, node.release());
       return;
   }
   ```
   - If `next` is `nullptr`, we attempt to attach the new node.
   - If successful, we update `tail_`.

### **Why is `continue` Necessary?**


Without `continue`, after `tail_.compare_exchange_weak(old_tail, next)`, the function would **proceed** without restarting the loop. This could result in:
- **Using stale values** of `old_tail` and `next`, potentially leading to incorrect behavior.
- **A missed enqueue operation**, as we wouldn’t retry with the new tail node. Which means that the `tail_` is not moved forward.

### **Summary**
The `continue` ensures that if `tail_` was not pointing to the correct node (due to concurrent updates), the loop **retries from the beginning** with updated values. This is critical for maintaining the correctness of the **lock-free queue** and avoiding race conditions.



## Correct vs. incorrect deque
Correct:
```cpp
bool deque(int& result) {
    SimpleNode* oldHead;
    SimpleNode* next;
    
    do {
        oldHead = head_.load();
        SimpleNode* oldTail = tail_.load();
        next = oldHead->next.load();

        if (oldHead != head_.load()) {
            continue; // Restart loop if head changed
        }

        if (oldHead == oldTail) { 
            if (next == nullptr) {
                return false; // Queue is empty
            }
            // Help advance the tail if it's stuck (optional optimization)
            tail_.compare_exchange_weak(oldTail, next);
        }
        
    } while (!head_.compare_exchange_weak(oldHead, next));

    // Ensure next is valid before accessing it
    if (next != nullptr) {
        result = std::move(next->val);
        delete oldHead;
        return true;
    }

    return false; // Failsafe check, should not be reached
}
```

Incorrect:
```cpp
  bool deque(int& val) {
    SimpleNode* old_head = nullptr;
    SimpleNode* old_tail = nullptr;
    SimpleNode* next = nullptr;
    do {
      old_head = head_.load();
      old_tail = tail_.load();
      assert(old_head != nullptr);
      next = old_head->next.load();
      if(old_head != head_.load()) {
        continue;
      }
      if(old_head == old_tail && next == nullptr) {
        return false;
      }
    } while(head_.compare_exchange_weak(old_head, next) == false);

    val = next->val;
    delete old_head;
    return true;
  }
```

Explanation from gpt. Basically it means that we need to whether next is nullptr or not.
Your `deque` function has a **bug that can cause a segmentation fault (crash)** due to **dereferencing a null pointer (`next`)** when the queue is empty. Here's what happens:

### **Identified Bugs:**
1. **Null Dereference of `next`**
   - When `oldHead == oldTail` and `next == nullptr`, the queue is empty.  
   - However, the loop continues, and `next` (which is `nullptr`) is used in `head_.compare_exchange_weak(oldHead, next)`, leading to **undefined behavior**.
   - Also, `result = std::move(next->val);` will crash if `next` is `nullptr`.

### **Fixed Code**
```cpp
bool deque(int& result) {
    SimpleNode* oldHead;
    SimpleNode* next;
    
    do {
        oldHead = head_.load();
        SimpleNode* oldTail = tail_.load();
        next = oldHead->next.load();

        if (oldHead != head_.load()) {
            continue; // Restart loop if head changed
        }

        if (oldHead == oldTail) { 
            if (next == nullptr) {
                return false; // Queue is empty
            }
            // Help advance the tail if it's stuck (optional optimization)
            tail_.compare_exchange_weak(oldTail, next);
        }
        
    } while (!head_.compare_exchange_weak(oldHead, next));

    // Ensure next is valid before accessing it
    if (next != nullptr) {
        result = std::move(next->val);
        delete oldHead;
        return true;
    }

    return false; // Failsafe check, should not be reached
}
```

### **Fixes & Improvements**
1. **Ensured `next` is not null before accessing it**
   - We only proceed if `next` is valid.
2. **Added a tail advancement step** *(optional but improves performance)*
   - If `tail_` is outdated (`oldTail == oldHead && next != nullptr`), update it to avoid slow tail progression.
3. **Handled the empty queue case properly**
   - Now correctly returns `false` when the queue is empty without further processing.

This prevents **segmentation faults**, ensures correct behavior, and slightly optimizes the tail update. 🚀

## The difference between two enque implementation
First one, correct

```cpp
  void enque(int val) {
    std::unique_ptr<SimpleNode> node(new SimpleNode());
    node->val = val;
    SimpleNode* old_tail = nullptr;
    SimpleNode* next  = nullptr;  
    while(true) {
      old_tail = tail_.load();
      next = old_tail->next.load();
      if(old_tail != tail_.load()) {
        continue;
      
      }
      if(next != nullptr) {
        tail_.compare_exchange_weak(old_tail, next);
        continue;
      }

      if(old_tail->next.compare_exchange_weak(next, node.get())) {

      tail_.compare_exchange_weak(old_tail, node.release());
        return;
      }
    } 
  }
```

Second one, incorrect
```cpp
 void enque(int val) {
        SimpleNode* node = new SimpleNode(val);
        SimpleNode* old_tail;
        SimpleNode* next;

        do {
            old_tail = tail_.load();
            next = old_tail->next.load();
            if (old_tail != tail_.load()) {
                continue;
            }

            if (next != nullptr) {
                tail_.compare_exchange_weak(old_tail, next);
                continue;
            }
        } while (!old_tail->next.compare_exchange_weak(next, node));

        tail_.compare_exchange_weak(old_tail, node);
        size_.fetch_add(1, std::memory_order_relaxed);
    }
```


### **Key Difference: When `tail_` is Updated**
You're correct that **both implementations update `tail_` only once when successfully adding a node**. However, **the crucial difference is what happens when `next != nullptr`** (i.e., when another thread is already modifying the queue).

### **First Implementation (Correct)**
```cpp
if (next != nullptr) {
    tail_.compare_exchange_weak(old_tail, next);
    continue;
}
```
🔹 **If another thread is already enqueueing a node (`next != nullptr`), we help move `tail_` forward before retrying.**  
🔹 This prevents `tail_` from becoming stale and ensures it always points to the latest node.

### **Second Implementation (Incorrect)**
```cpp
if (next != nullptr) {
    tail_.compare_exchange_weak(old_tail, next);
    continue;
}
```
🔹 **This part is identical**, but **the retry loop does not guarantee that `tail_` is updated correctly at the end**.  
🔹 Since `tail_.compare_exchange_weak(old_tail, node);` is **only executed once, outside the loop**, it might miss updates when multiple threads are enqueueing.

---

## **Example: The Core Issue in the Second Implementation**
Let's break it down step by step **with two threads (`Thread 1` and `Thread 2`) enqueuing at the same time**.

### **Initial Queue State**
```
HEAD -> Node(A) -> (TAIL)
```
Both `head_` and `tail_` point to `Node(A)`.  
Now, **Thread 1 enqueues `Node(B)`, and Thread 2 enqueues `Node(C)`.**

---

### **Step 1: Both Threads Read `tail_`**
- **Thread 1** reads `tail_ = Node(A)`, sees `next = nullptr`, and creates `Node(B)`.
- **Thread 2** reads `tail_ = Node(A)`, sees `next = nullptr`, and creates `Node(C)`.

---

### **Step 2: Thread 1 Successfully Adds `Node(B)`**
- **Thread 1 succeeds** in:
  ```cpp
  old_tail->next.compare_exchange_weak(next, node_B);
  ```
  - Now, the queue looks like:
    ```
    HEAD -> Node(A) -> Node(B)
    ```
- **Thread 2 retries because `next` is no longer `nullptr`!**

---

### **Step 3: Thread 1 Updates `tail_`**
- **Thread 1 successfully updates `tail_` to point to `Node(B)`.**
  ```
  HEAD -> Node(A) -> Node(B) -> (TAIL)
  ```

---

### **Step 4: Thread 2 Retries**
- **Thread 2 now reads `tail_ = Node(B)`, next = `nullptr`, and successfully adds `Node(C)`.**
  ```
  HEAD -> Node(A) -> Node(B) -> Node(C)
  ```
- **But here’s the bug in the second implementation:**  
  - `tail_` is updated **only once, outside the loop**.
  - If another thread starts enqueueing at the same time, `tail_` **might still point to `Node(B)` instead of `Node(C)`!**
  - **Future enqueue operations might have to retry unnecessarily because `tail_` is stale.**

---

### **Step 5: The Problem (Stale `tail_`)**
- If a new thread tries to enqueue `Node(D)`, it **may still see `tail_ = Node(B)`, instead of `Node(C)`.**
- This forces unnecessary retries, slowing down the queue.

---

## **Why the First Implementation Avoids This**
In the **first implementation**, if another thread is already in the middle of an enqueue (i.e., `next != nullptr`), it **helps move `tail_` forward before retrying.**

### **Why This Matters**
Imagine `Thread 1` enqueues `Node(B)`, and `Thread 2` enqueues `Node(C)`.  
- **In the first implementation, `Thread 2` would help move `tail_` to `Node(B)` before proceeding.**  
- **This guarantees that when it enqueues `Node(C)`, `tail_` will be updated correctly.**  

By continuously ensuring that `tail_` is moved forward **before retrying**, the first implementation guarantees that `tail_` **always** points to the most recently enqueued node.

---

## **Summary of the Core Difference**
| Feature | **First Implementation (Correct)** | **Second Implementation (Incorrect)** |
|---------|----------------------------------|----------------------------------|
| **What happens when `next != nullptr`?** | Moves `tail_` forward before retrying. | Just retries without ensuring `tail_` is updated correctly. |
| **Ensures `tail_` always points to the latest node?** | ✅ Yes | ❌ No, `tail_` might point to an old node. |
| **Potential issue?** | No inefficiencies, always correct. | `tail_` can become stale, causing unnecessary retries. |
| **Performance impact** | More efficient, fewer retries. | Less efficient, more retries needed. |

---

### **Final Takeaway**
**The key reason the first implementation is correct is that it ensures `tail_` is always moved forward before retrying.**  
Without this, the second implementation **allows `tail_` to become stale**, which slows down the queue by forcing unnecessary retries.

---

I still don't get it. Why the second implementation is incorrect ?

You're absolutely right that **other threads can help move `tail_` forward**—this is the whole idea behind a lock-free queue! However, the second implementation **still has a correctness issue** because it does not ensure that `tail_` is updated reliably **before finishing the enqueue operation**.  

Let’s clarify **why this is a problem even though other threads can help move `tail_` forward.**  

---

## **Key Question**  
Since other threads can update `tail_`, why is the second implementation still incorrect?  

### **Answer:**  
Even though other threads **may** update `tail_`, the second implementation **relies on them to fix the issue rather than guaranteeing correctness itself.**  
This creates a **non-deterministic behavior** where `tail_` **might or might not** point to the latest node, depending on thread timing.

---

## **Breaking It Down Step by Step**
Let’s assume **three threads (`T1`, `T2`, and `T3`) are enqueuing at the same time** to see where the second implementation fails.

### **Initial State**
```
HEAD -> Node(A) (tail_)
```
- `head_` and `tail_` both point to `Node(A)`.
- Now three threads start enqueueing.

---

### **Step 1: T1 Successfully Enqueues `Node(B)`**
- `T1` successfully attaches `Node(B)`:
  ```
  HEAD -> Node(A) -> Node(B)
  ```
- But `tail_` is **still pointing to `Node(A)`**.
- Before `T1` can update `tail_`, **T2 and T3 start running.**

---

### **Step 2: T2 Reads Stale `tail_ = Node(A)`, Tries to Enqueue `Node(C)`**
- `T2` reads:
  ```
  old_tail = Node(A), next = Node(B)
  ```
- Since `next != nullptr`, **it helps move `tail_` to `Node(B)`, then retries.**
- **Now `tail_ = Node(B)`.**

---

### **Step 3: T2 Successfully Enqueues `Node(C)`**
- `T2` now enqueues `Node(C)`:
  ```
  HEAD -> Node(A) -> Node(B) -> Node(C)
  ```
- But `tail_` **is still pointing to `Node(B)`**.
- Before `T2` updates `tail_`, **T3 starts running.**

---

### **Step 4: T3 Reads Stale `tail_ = Node(B)`, Tries to Enqueue `Node(D)`**
- `T3` reads:
  ```
  old_tail = Node(B), next = Node(C)
  ```
- Since `next != nullptr`, **it helps move `tail_` to `Node(C)`, then retries.**
- **Now `tail_ = Node(C)`.**

---

### **Step 5: T3 Successfully Enqueues `Node(D)`**
- `T3` enqueues `Node(D)`:
  ```
  HEAD -> Node(A) -> Node(B) -> Node(C) -> Node(D)
  ```
- But `tail_` is **still pointing to `Node(C)`**.
- If **no other thread updates `tail_` now, it remains stale!**

---

## **The Core Problem in the Second Implementation**
If **all threads finish** and no new thread is enqueueing, `tail_` **might be stuck at an old node** (e.g., `Node(C)` instead of `Node(D)`).  

This **isn't guaranteed to cause a failure immediately**, but it creates two major issues:  

1. **Future threads must do extra retries to move `tail_` forward.**  
   - If `tail_` lags behind, future enqueue operations will see `next != nullptr` and must **repeatedly help move `tail_` forward before enqueuing.**
   - This **adds unnecessary contention and performance overhead.**

2. **If a dequeue operation runs now, it may have to traverse multiple nodes before reaching `tail_`.**  
   - A correct `tail_` should always point to the last node, minimizing the traversal cost for future operations.

### **Why Doesn't This Happen in the First Implementation?**
The **first implementation guarantees that `tail_` always points to the latest node before returning**:
```cpp
if (next != nullptr) {
    tail_.compare_exchange_weak(old_tail, next);
    continue;
}
```
- If `next != nullptr`, we **keep updating `tail_` and retrying until we attach our node to the latest position**.
- This ensures that when the function exits, `tail_` **always points to the correct last node**.

---

## **Final Takeaway**
🔹 **Yes, other threads can update `tail_`, but the second implementation relies on this happening rather than guaranteeing correctness itself.**  
🔹 **If no other thread updates `tail_`, `tail_` can become stale, leading to inefficiencies and unnecessary retries.**  
🔹 The **first implementation actively ensures `tail_` is always updated before returning**, making it more robust and efficient.  


I still don't get it.  I will just give up then.
At least I learn what's the correct implementaion of lock free queue.



## Full test code 
```cpp
#include <iostream>
#include <thread>
#include <cassert>
#include <atomic>


using namespace std;

struct Node{
  Node(int v) {
    val = v;
    next = prev = nullptr;
  }
  int val;
  Node* next;
  Node* prev;
};

class Queue {
public:
  Queue() {
    head_ = tail_ = new Node(0);
    size_ = 0;
    head_->next = tail_;
    tail_->prev = head_;


  }

  void enque(int val) {
    Node* node = new Node(val);
    node->next = tail_;
    node->prev = tail_->prev;
    tail_->prev->next = node;
    tail_->prev = node;
    size_++;


  }

  bool deque(int& val) {
    if (size_ == 0) {
      return false;
    }
    Node* node = head_->next;
    val = node->val;
    head_->next = node->next;
    node->next->prev = head_;
    delete node;
    size_--;
    return true;

  }

  bool empty() {
    return size_ == 0;

  }

  int front() {
    return head_->next->val;

  }


private:
  Node* head_;
  Node* tail_;
  int size_;


};


struct SimpleNode {
  int val;
  std::atomic<SimpleNode*> next;
  SimpleNode() {
    val = 0;
    next = nullptr;
  }
  SimpleNode(int v) {
    val = v;
    next = nullptr;
  }

};


class MyLockFreeQueue {
public:
  MyLockFreeQueue()  {
    SimpleNode* node = new SimpleNode();
    node->next = nullptr;
    head_.store(node);
    tail_.store(node);

  }
// void enque(int val) {
//     // Allocate the new node
//     std::unique_ptr<SimpleNode> node = std::make_unique<SimpleNode>(val);

//     SimpleNode* old_tail = nullptr;
//     SimpleNode* next = nullptr;

//     while (true) {
//         old_tail = tail_.load();
//         next = old_tail->next.load();

//         if (old_tail != tail_.load()) {
//             continue; // Retry if tail changed
//         }

//         if (next != nullptr) {
//             // Help advance tail if it's stale
//             tail_.compare_exchange_weak(old_tail, next);
//             continue;
//         }

//         // Try to attach the new node
//         if (old_tail->next.compare_exchange_weak(next, node.get())) {
//             // Successfully linked, now update tail_
//             tail_.compare_exchange_weak(old_tail, node.release());
//             return;
//         }
//     }
// }

  void enque(int val) {
    std::unique_ptr<SimpleNode> node(new SimpleNode());
    node->val = val;
    SimpleNode* old_tail = nullptr;
    SimpleNode* next  = nullptr;  
    while(true) {
      old_tail = tail_.load();
      next = old_tail->next.load();
      if(old_tail != tail_.load()) {
        continue;
      
      }
      if(next != nullptr) {
        tail_.compare_exchange_weak(old_tail, next);
        continue;
      }

      if(old_tail->next.compare_exchange_weak(next, node.get())) {

      tail_.compare_exchange_weak(old_tail, node.release());
        return;
      }
    } 
  }

    int size() const {
    SimpleNode* node = head_.load();
    assert(node != nullptr);
    int count = 0;
    while(node != tail_.load()) {
      count++;
      if(node == nullptr) {
        cout << "node is null, count is " << count<< endl;

      }
      assert(node != nullptr);
      node = node->next.load();
    }
    return count;
  }

  // bool deque(int& val) {
  //   SimpleNode* old_head = nullptr;
  //   SimpleNode* old_tail = nullptr;
  //   SimpleNode* next = nullptr;
  //   do {
  //     old_head = head_.load();
  //     old_tail = tail_.load();
  //     assert(old_head != nullptr);
  //     next = old_head->next.load();
  //     if(old_head != head_.load()) {
  //       continue;
  //     }
  //     if(old_head == old_tail && next == nullptr) {
  //       return false;
  //     }
  //   } while(head_.compare_exchange_weak(old_head, next) == false);

  //   val = next->val;
  //   delete old_head;
  //   return true;
  // }

bool deque(int& result) {
    SimpleNode* oldHead;
    SimpleNode* next;
    
    while (true) {
        oldHead = head_.load();
        SimpleNode* oldTail = tail_.load();
        next = oldHead->next.load();

        if (oldHead != head_.load()) {
            continue; // Restart loop if head changed
        }

        if (oldHead == oldTail) { 
            if (next == nullptr) {
                return false; // Queue is empty
            }
            // Help advance the tail if it's stuck (optional optimization)
            tail_.compare_exchange_weak(oldTail, next);
        continue;
        }
        if(head_.compare_exchange_weak(oldHead, next)) {
          // Ensure next is valid before accessing it
          if (next != nullptr) {
              result = std::move(next->val);
              delete oldHead;
              return true;
          }


          }
        
    } 

    return false; // Failsafe check, should not be reached
}

    // bool deque(int& result) {
    //     SimpleNode* oldHead;
    //     SimpleNode* next;
    //     do {
    //         oldHead = head_.load();
    //         SimpleNode* oldTail = tail_.load();
    //         next = oldHead->next.load();
    //         if (oldHead!= head_.load()) {
    //             continue;
    //         }
    //         if (oldHead == oldTail && next == nullptr) {
    //             return false;
    //         }
    //     } while (!head_.compare_exchange_weak(oldHead, next));
    //     result = std::move(next->val);
    //     delete oldHead;
    //     return true;
    // }


  bool empty() {
    return head_.load()->next.load() == nullptr;
  }


  private:
  // size
  std::atomic<int> size_;
  std::atomic<SimpleNode*> head_;
  std::atomic<SimpleNode*> tail_;
};

#include <atomic>

class MyLockFreeQueue2 {
public:
    MyLockFreeQueue2() : size_(0) {
        SimpleNode* node = new SimpleNode();
        head_ = tail_ = node;
    }

    void enque(int val) {
        SimpleNode* node = new SimpleNode(val);
        SimpleNode* old_tail;
        SimpleNode* next;

        do {
            old_tail = tail_.load();
            next = old_tail->next.load();
            if (old_tail != tail_.load()) {
                continue;
            }

            if (next != nullptr) {
                tail_.compare_exchange_weak(old_tail, next);
                continue;
            }
        } while (!old_tail->next.compare_exchange_weak(next, node));

        tail_.compare_exchange_weak(old_tail, node);
        size_.fetch_add(1, std::memory_order_relaxed);
    }

    bool deque(int& val) {
        SimpleNode* old_head;
        SimpleNode* old_tail;
        SimpleNode* next;

        do {
            old_head = head_.load();
            old_tail = tail_.load();
            next = old_head->next.load();

            if (old_head != head_.load()) {
                continue;
            }

            if (old_head == old_tail) {
                if (next == nullptr) {
                    return false;
                }
                // tail_.compare_exchange_weak(old_tail, next);
                continue;
            }

            val = next->val;
        } while (!head_.compare_exchange_weak(old_head, next));

        safe_delete(old_head); // Use a safe delete method
        size_.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    bool empty() {
        return head_.load()->next.load() == nullptr;
    }

  int size() {
    return size_.load();
  }


private:
    std::atomic<int> size_;
    std::atomic<SimpleNode*> head_;
    std::atomic<SimpleNode*> tail_;

    void safe_delete(SimpleNode* node) {
    delete node;
        // Implement a safe deletion mechanism to prevent use-after-free issues
        // This could be done with a garbage collector or hazard pointers.
    }
};

class LockFreeQueue{

public:
  LockFreeQueue() {
    head_ = tail_ = new Node(0);
    head_->next = tail_;
    tail_->prev = head_;
  }

  void enque(int val) {
    Node* node = new Node(val);
    Node* prev = tail_->prev;
    while (true) {
      Node* next = prev->next;
      node->next = next;
      node->prev = prev;
      if (__sync_bool_compare_and_swap(&prev->next, next, node)) {
        next->prev = node;
        break;
      }
    }
  }

  bool deque(int& val) {
    Node* node;
    while (true) {
      node = head_->next;
      if (node == tail_) {
        return false;
      }
      Node* next = node->next;
      if (__sync_bool_compare_and_swap(&head_->next, node, next)) {
        next->prev = head_;
        break;
      }
    }
    val = node->val;
    delete node;
    return true;
  }

  bool empty() {
    return head_->next == tail_;
  }

  int front() {
    return head_->next->val;
  }
private:
  Node* head_;
  Node* tail_;
};

void test_my_lock_free_queue() {
  // enque 100 times concurrently and dequeue 100 times concurrently
  MyLockFreeQueue q;
  const int num_threads = 10;
  const int num_ops = 100;
  std::thread threads[num_threads];
  std::thread dequeue_threads[num_threads];
  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread([&q, i, num_ops] {
      for (int j = 0; j < num_ops; j++) {
        q.enque(i * num_ops + j);
      }
    });
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
  cout << " hello " << endl;
  // get queue size
  cout << "size " << q.size() << endl;


  // return;
  for (int i = 0; i < num_threads; i++) {
    dequeue_threads[i] = std::thread([&q, i, num_ops] {
      for (int j = 0; j < num_ops; j++) {
        int val;
        bool res =  q.deque(val);
        // cout << "cur val " << val << endl;
        if(!res) {
          cout << "cur num " << i << " cur ops " << j << endl;
          assert(res);
        }
      }
    });
  }
  for (int i = 0; i < num_threads; i++) {
    dequeue_threads[i].join();
  }
  assert(q.empty());
  cout << "All tests pass" << endl;

}

void testnormalqueue() {
Queue q;
  q.enque(1);
  q.enque(2);
  q.enque(3);
  int val;
  q.deque(val);
  q.enque(4);
  assert(q.front() == 2);
  q.deque(val);
  assert(q.front() == 3);
  q.deque(val);
  assert(q.front() == 4);
  q.deque(val);
  assert(q.empty());
  cout << "All tests pass" << endl;

}
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <cassert>
#include <mutex>

void test_concurrent_lock_free_queue() {
    MyLockFreeQueue q;
    const int num_threads = 10;
    const int num_ops = 100;

    std::thread enqueue_threads[num_threads];
    std::thread dequeue_threads[num_threads];

    // Enqueue and dequeue concurrently
    for (int i = 0; i < num_threads; i++) {
        enqueue_threads[i] = std::thread([&q, i, num_ops] {
            for (int j = 0; j < num_ops; j++) {
                q.enque(i * num_ops + j);
            }
        });

        dequeue_threads[i] = std::thread([&q, i, num_ops] {
            for (int j = 0; j < num_ops; j++) {
                int val;
                if (q.deque(val)) { // Check return value to avoid undefined behavior
                    // Process val (optional)
                }
            }
        });
    }

    // Join all threads
    for (int i = 0; i < num_threads; i++) {
        enqueue_threads[i].join();
        dequeue_threads[i].join();
    }

    // Ensure queue is empty
    // assert(q.empty());

    static std::mutex cout_mutex;
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << q.size() << std::endl;
        std::cout << "test concurrent lock free queue All tests pass" << std::endl;
    }
}

int main() {
  // test_concurrent_lock_free_queue();
  test_my_lock_free_queue();
    return 0;
}
```


## References
[https://book-of-gehn.github.io/articles/2020/04/28/Lock-Free-Queue-Part-II.html](https://book-of-gehn.github.io/articles/2020/04/28/Lock-Free-Queue-Part-II.html)
I don't fully get the idea in this article.

[https://mp.weixin.qq.com/s?__biz=Mzg4NDQ0OTI4Ng==&mid=2247490953&idx=1&sn=00dd064b978d2bae85939f5e387d1022&chksm=cfb954e0f8ceddf641401126ad96bb5ca8ec37d206f66792aaae9f48decdc1292eef2c296f16&cur_album_id=3140091333123276802&scene=189#wechat_redirect](https://mp.weixin.qq.com/s?__biz=Mzg4NDQ0OTI4Ng==&mid=2247490953&idx=1&sn=00dd064b978d2bae85939f5e387d1022&chksm=cfb954e0f8ceddf641401126ad96bb5ca8ec37d206f66792aaae9f48decdc1292eef2c296f16&cur_album_id=3140091333123276802&scene=189#wechat_redirect)
This article talks about licked list implementaion and array implementaion which is useful.


[https://github.com/cameron314/concurrentqueue](https://github.com/cameron314/concurrentqueue)
