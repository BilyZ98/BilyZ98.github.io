# C++

## emplace() in unordered_map 
The `emplace()` function is a member function of the `std::unordered_map` container in C++. It is used to insert a new element into the map. The main advantage of `emplace()` over the `insert()` function is that it constructs the element **in-place**, thus avoiding unnecessary copy or move operations.

Here is the syntax for `emplace()`:

```cpp
pair<iterator, bool> emplace (const value_type& val);
```

It takes a `value_type` object as an argument, which is a `std::pair` containing the key and value to be inserted. The function returns a pair, where `pair::first` is an iterator pointing to the element (whether newly inserted or already present), and `pair::second` is a boolean denoting whether the insertion took place (true if new element was inserted, false if element already existed).

Here's an example of how to use `emplace()`:

```cpp
#include <iostream>
#include <unordered_map>

int main() {
    std::unordered_map<std::string, int> my_map;

    // Using emplace() to insert elements
    auto res = my_map.emplace("apple", 1);
    if (res.second) {
        std::cout << "Insertion successful, " << res.first->first << " => " << res.first->second << '\n';
    }

    res = my_map.emplace("apple", 2);
    if (!res.second) {
        std::cout << "Element with key 'apple' already exists with value " << res.first->second << '\n';
    }

    return 0;
}
```

In this code, we first attempt to insert the pair ("apple", 1) into `my_map`. Since the map is initially empty, the insertion is successful. The second attempt to insert ("apple", 2) fails because a pair with the key "apple" already exists in the map. The `emplace()` function returns a pair, where the first element is an iterator to the existing element, and the second element is a boolean that is `false` because the insertion did not take place. The existing value of "apple" is then printed. 

This example demonstrates the efficiency of `emplace()`, as it avoids creating a temporary pair object when inserting an element into the map. Instead, the pair is constructed in-place. This can lead to performance improvements, especially for large objects. 

Remember, `emplace()` is a part of C++11 and later. If you're using an older version of C++, you might not have access to this function. 


## Difference between reinterpret_cast and static_cast
`static_cast` and `reinterpret_cast` are two types of casting operators in C++. They are used to convert one type to another, but they serve different purposes and have different restrictions:

- **static_cast**: This is the most commonly used cast and is generally considered safer than `reinterpret_cast`. It performs conversions between types that are related by inheritance (i.e., between a base class and a derived class) or standard conversions (e.g., from `int` to `float`). It does not perform checks for type compatibility¹².

- **reinterpret_cast**: This cast converts any pointer type to any other pointer type, even unrelated classes. It can also cast pointers to or from integer types. The operation result is a simple binary copy of the value from one pointer to the other. It does not check if the pointed-to types are the same, and using the resulting pointer can lead to undefined behavior¹².

Here's a key difference: `static_cast` is more restrictive and safer than `reinterpret_cast`. `static_cast` only allows conversions that could under some circumstances succeed and be meaningful, such as `int` to `float` or base class pointer to derived class pointer. On the other hand, `reinterpret_cast` allows any conversion, even those that are likely to be meaningless or unsafe³.

In general, you should always prefer `static_cast` for casting that should be safe. If you accidentally try doing a cast that isn't well-defined, then the compiler will report an error¹. `reinterpret_cast` should be used sparingly and carefully, as it can lead to unsafe or undefined behavior².

## Encode float number and write to file 
```cpp
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <type_traits>


bool isLittleEndian() {
  int x = 1;
  return *(char*)&x;
}


inline void EncodeFixed32(char* buf, uint32_t value) {
  if (isLittleEndian()) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
  }
}

inline void PutFixed32(std::ofstream* dst, uint32_t value) {
  if (isLittleEndian()) {

    dst->write(const_cast<const char*>(reinterpret_cast<char*>(&value)),
                sizeof(value));
  } else {
    char buf[sizeof(value)];
    EncodeFixed32(buf, value);
    dst->write(buf, sizeof(buf));
  }
}

inline uint32_t DecodeFixed32(const char* ptr) {
  if (isLittleEndian()) {
    std::cout << "little endian" << std::endl;
    // Load the raw bytes
    uint32_t result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint32_t>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}
inline bool GetFixed32(std::string* input, uint32_t* value) {
  if (input->size() < sizeof(uint32_t)) {
    return false;
  }
  *value = DecodeFixed32(input->data());
  // input->remove_prefix(sizeof(uint32_t));
  return true;
}
int main() {
    // open file to write
  std::ofstream file("test.txt", std::ios::binary);
  if (file.is_open()) {
    float f = 3.14;
    PutFixed32(&file, *reinterpret_cast<uint32_t*>(&f));
    file.close();

  }

  // open file to read
  std::ifstream file2("test.txt", std::ios::binary);
  if (file2.is_open()) {
    std::string line;
    uint32_t value;
    if (std::getline(file2, line)) {
      GetFixed32(&line, &value);
      float f = *reinterpret_cast<float*>(&value);
      std::cout << value << std::endl;

      std::cout << f << std::endl;
    }
  }


    return 0;
}
```

Output: 
```bash
(base) ➜  /tmp ./a.out
little endian
1078523331
3.14

```

```cpp

#include <fstream>

int main() {
    std::ofstream outfile("float.bin", std::ios::binary);
    float pi = 3.14159;
    outfile.write(reinterpret_cast<char*>(&pi), sizeof(pi));
    outfile.close();
    return 0;
}
```

```cpp
inline void EncodeFloat32(char* buf, float value) {
  uint32_t int_value = reinterpret_cast<uint32_t&>(value);
  if (port::kLittleEndian) {
    memcpy(buf, &int_value, sizeof(int_value));
  } else {
    buf[0] = int_value & 0xff;
    buf[1] = (int_value >> 8) & 0xff;
    buf[2] = (int_value >> 16) & 0xff;
    buf[3] = (int_value >> 24) & 0xff;
  }
}

inline float DecodeFloat32(const char* buf) {
  uint32_t int_value;
  if (port::kLittleEndian) {
    memcpy(&int_value, buf, sizeof(int_value));
  } else {
    int_value = (static_cast<uint32_t>(buf[0]) |
                 (static_cast<uint32_t>(buf[1]) << 8) |
                 (static_cast<uint32_t>(buf[2]) << 16) |
                 (static_cast<uint32_t>(buf[3]) << 24));
  }
  return reinterpret_cast<float&>(int_value);
}

```



## Random sample among unordered_map 
To get a random sample from an `unordered_map` in C++, you can use an iterator and the `std::advance` function. Here's an example:

```cpp
#include <unordered_map>
#include <cstdlib> // for rand()
#include <iterator> // for std::advance

std::unordered_map<std::string, int> my_map;
// Assume my_map is populated...

// Generate a random index
int random_index = rand() % my_map.size();

// Create an iterator pointing to the start of the map
auto it = my_map.begin();

// Advance the iterator by the random index
std::advance(it, random_index);

// Now, 'it' points to a random element in the map
std::string random_key = it->first;
int random_value = it->second;
```

In this code, `rand() % my_map.size()` generates a random index. `std::advance(it, random_index)` moves the iterator `it` forward by `random_index` steps³. Please note that this method involves linear time complexity. If you need to do this operation frequently and performance is a concern, you might want to consider a different data structure or approach. 

Also, don't forget to seed the random number generator using `srand(time(0))` at the beginning of your program to ensure you get different results each time you run your program. If you don't seed the random number generator, `rand()` will return the same sequence of numbers every time you run your program. However, `rand()` and `srand()` are considered somewhat outdated, and a more modern approach is to use the random library introduced in C++11. Here's how you can do it:

```cpp
#include <unordered_map>
#include <random> // for std::random_device and std::mt19937
#include <iterator> // for std::advance

std::unordered_map<std::string, int> my_map;
// Assume my_map is populated...

// Create a random number generator
std::random_device rd;
std::mt19937 gen(rd());

// Create a uniform distribution from 0 to my_map.size() - 1
std::uniform_int_distribution<> distr(0, my_map.size() - 1);

// Generate a random index
int random_index = distr(gen);

// Create an iterator pointing to the start of the map
auto it = my_map.begin();

// Advance the iterator by the random index
std::advance(it, random_index);

// Now, 'it' points to a random element in the map
std::string random_key = it->first;
int random_value = it->second;
```

In this version, `std::random_device` is a uniformly-distributed integer random number generator that produces non-deterministic random numbers. `std::mt19937` is a Mersenne Twister random number generator. `std::uniform_int_distribution` produces random integers in a range [a, b], where each possible number has an equal chance of being produced³.

Source: Conversation with Bing, 2/20/2024
(1) Select random element in an unordered_map - Stack Overflow. https://stackoverflow.com/questions/27024269/select-random-element-in-an-unordered-map.
(2) std::unordered_map - cppreference.com. https://en.cppreference.com/w/cpp/container/unordered_map.
(3) unordered_map in C++ STL - GeeksforGeeks. https://www.geeksforgeeks.org/unordered_map-in-cpp-stl/.

## Concurrent unordered_map in cpp
In C++, the `std::unordered_map` is not thread-safe¹. Concurrent updates to an `unordered_map` can lead to data races and undefined behavior. 

If you need to perform concurrent updates, you might want to consider using a thread-safe container. For example, Intel's Threading Building Blocks (TBB) library provides `tbb::concurrent_hash_map` and `tbb::concurrent_unordered_map`². These containers support fine-grained, per-key locking for insert/update operations².

Microsoft's Parallel Patterns Library (PPL) also provides a `concurrent_unordered_map` class³. This class is a concurrency-safe container that enables concurrency-safe append, element access, iterator access, and iterator traversal operations³.

Here's a simple example of how you might use `tbb::concurrent_hash_map`:

```cpp
#include <tbb/concurrent_hash_map.h>
#include <iostream>

int main() {
    tbb::concurrent_hash_map<int, int> map;

    // Insert a key-value pair into the map
    map.insert(std::make_pair(1, 100));

    // Update a value in the map
    tbb::concurrent_hash_map<int, int>::accessor accessor;
    if (map.find(accessor, 1)) {
        accessor->second = 200;
    }

    // Print the value of key 1
    std::cout << "Value of key 1: " << map.find(accessor, 1)->second << std::endl;

    return 0;
}
```

This code first inserts a key-value pair `(1, 100)` into the map. It then updates the value associated with key `1` to `200`. Finally, it prints the value associated with key `1`, which is `200`.

Remember to link against the TBB library when compiling this code.

Please note that you should always handle synchronization carefully when dealing with concurrent updates to avoid potential issues such as deadlocks and race conditions. If possible, design your program in a way that minimizes the need for shared mutable state.

Source: Conversation with Bing, 2/16/2024
(1) std::unordered_map - cppreference.com. https://en.cppreference.com/w/cpp/container/unordered_map.
(2) c++ - unordered_map thread safety - Stack Overflow. https://stackoverflow.com/questions/9685486/unordered-map-thread-safety.
(3) concurrent_unordered_map Class | Microsoft Learn. https://learn.microsoft.com/en-us/cpp/parallel/concrt/reference/concurrent-unordered-map-class?view=msvc-170.
(4) Parallel Containers and Objects | Microsoft Learn. https://learn.microsoft.com/en-us/cpp/parallel/concrt/parallel-containers-and-objects?view=msvc-170.

## std::shared_ptr
`std::shared_ptr` is a smart pointer in the C++ Standard Library that retains shared ownership of an object through a pointer. Multiple `shared_ptr` objects may own the same object, and the object is destroyed when the last `shared_ptr` is destroyed.

Here's a basic example of how to use `std::shared_ptr`:

```cpp
#include <memory>

struct Foo {
    Foo() { std::cout << "Foo...\n"; }
    ~Foo() { std::cout << "~Foo...\n"; }
};

int main() {
    std::shared_ptr<Foo> sp1(new Foo);  // Foo...
    {
        std::shared_ptr<Foo> sp2 = sp1;
        // The destructor is not called here because sp1 also points to the object
    }
    // The destructor is called here because sp1 goes out of scope and no other shared_ptr points to the object
    return 0;
}
```

In this example, `sp1` and `sp2` share ownership of the same `Foo` object. The `Foo` object is only destroyed when both `sp1` and `sp2` have been destroyed, ensuring that the object's lifetime lasts as long as there is a `shared_ptr` pointing to it. This is the key feature of `std::shared_ptr`.

Remember, it's generally a good practice to use `std::make_shared` instead of `new` when creating a `shared_ptr`, because `std::make_shared` combines the allocation of the control block and the managed object into a single memory allocation, which can improve performance. Here's how you can do it:

```cpp
std::shared_ptr<Foo> sp = std::make_shared<Foo>();
```

Please note that `std::shared_ptr` uses reference counting, which means that it keeps track of how many `shared_ptr`s point to the same object. This can introduce a slight overhead, but ensures that the object is properly deallocated when no longer in use. However, `std::shared_ptr` cannot resolve circular references. If you have two objects that point to each other using `shared_ptr`, you can have a memory leak. In such cases, consider using `std::weak_ptr`.

shared_ptr with customed deletion function
```
#include <memory>
#include <iostream>

struct Foo {
    Foo() { std::cout << "Foo...\n"; }
    ~Foo() { std::cout << "~Foo...\n"; }
};

void customDeleter(Foo* ptr) {
    std::cout << "Custom deleter...\n";
    delete ptr;
}

int main() {
    std::shared_ptr<Foo> sp(new Foo, customDeleter);
    return 0;
}
```

If you replace the pointer of a `std::shared_ptr` while other `std::shared_ptr`s are still pointing to it, the original object will not be deleted. This is because `std::shared_ptr` uses reference counting, and the count is not zero if there are still `std::shared_ptr`s pointing to the object.

Here's an example:

```cpp
#include <memory>
#include <iostream>

struct Foo {
    Foo() { std::cout << "Foo...\n"; }
    ~Foo() { std::cout << "~Foo...\n"; }
};

int main() {
    std::shared_ptr<Foo> sp1 = std::make_shared<Foo>();  // Foo...
    {
        std::shared_ptr<Foo> sp2 = sp1;
        sp1 = std::make_shared<Foo>();  // Foo...
        // The destructor is not called here because sp2 is still pointing to the first Foo object
    }
    // The destructor for the first Foo object is called here because sp2 goes out of scope
    // The destructor for the second Foo object is called here because sp1 goes out of scope
    return 0;
}
```

In this example, `sp1` and `sp2` initially point to the same `Foo` object. When we assign a new `std::shared_ptr` to `sp1`, `sp1` points to a new `Foo` object, but `sp2` still points to the original `Foo` object. The original `Foo` object is not destroyed until `sp2` is destroyed, which happens when `sp2` goes out of scope.

So, even if you replace the pointer of a `std::shared_ptr`, any other `std::shared_ptr`s pointing to the original object will still keep that object alive. This is one of the key features of `std::shared_ptr` and why it's useful for managing dynamically allocated memory.




## Static variable in cpp

Yes, there are several things you need to be careful about when dealing with static variables in C++ header files:

1. **Scope and Linkage**: Static variables have translation unit scope, usually a .c or .cpp file¹². An `#include` directive simply copies the text of a file verbatim and does not create another translation unit¹. Therefore, if you declare a static variable in a header file and include that header file in multiple source files, each source file will get its own copy of that static variable².

2. **Variable Conflicts**: If you declare a static variable with the same name in multiple header files and include these headers in the same source file, it will result in a conflict¹. This is because in a compilation unit (a .cpp file with the contents of the .h file inserted in place of each `#include` directive), you can't have two global variables with the same name¹.

3. **Best Practices**: It's generally not recommended to declare variables in header files, as this can often lead to subtle bugs and linker errors³. The effect of including a header file should be deterministic and idempotent⁵. That is, including it several times should have the same effect as including it once, and it should not have unnecessary or surprising effects⁵.

If you need to share a variable across multiple source files, consider using `extern` instead of `static`, and define the variable in one source file². If you're using C++, both `VAL` and `ANOTHER_VAL` are static².

Here's an example of how you might do this:

```cpp
// In header file
extern int shared_var;

// In one source file
int shared_var = 10;
```

In this example, `shared_var` is declared in the header file and defined in one source file. It can then be accessed from any source file that includes the header file. Remember to protect your header files with include guards to prevent multiple inclusion. 

### Why static variables?
Answer from Bing Chat
Static variables in programming languages like C and C++ serve several important purposes:

1. **Preserving State**: Static variables retain their value even when they go out of their scope¹². This means that static variables preserve their prior value in their previous scope and don't require reinitialization when entering a new scope¹². This is useful for cases where a function needs to keep some state between invocations².

2. **Lifetime**: A static variable remains in memory while the program is running¹. A normal or auto variable is destroyed when a function call where the variable was declared is over¹. For example, we can use a static variable to count the number of times a function is called¹.

3. **Memory Allocation**: Static variables are allocated memory in the data segment, not the stack segment¹. 

4. **Access Control**: In the C programming language, static is used with global variables and functions to set their scope to the containing file². This is used widely as an "access control" feature. If you have a .c file implementing some functionality, it usually exposes only a few "public" functions to users. The rest of its functions should be made static, so that the user won't be able to access them². This is encapsulation, a good practice².

Here's an example of how a static variable can preserve its state:

```cpp
#include <stdio.h>

void foo() {
    int a = 10;
    static int sa = 10;

    a += 5;
    sa += 5;

    printf("a = %d, sa = %d\\n", a, sa);
}

int main() {
    int i;

    for (i = 0; i < 10; ++i)
        foo();
}
```

This program prints:

```
a = 15, sa = 15
a = 15, sa = 20
a = 15, sa = 25
a = 15, sa = 30
a = 15, sa = 35
a = 15, sa = 40
a = 15, sa = 45
a = 15, sa = 50
a = 15, sa = 55
a = 15, sa = 60
```

As you can see, the static variable `sa` retains its value between function calls, while the non-static variable `a` does not².


### Static function in cpp class
In C++, a static function is a member function of a class that is associated with the class itself rather than with an instance or object of the class³. This means that a static function can be called without creating an instance of the class³. The `static` keyword is used to define a static function in C++³.

Here are some key properties of static functions in C++:

1. **Independence**: A static member function is independent of any object of the class¹.
2. **Accessibility**: A static member function can be called even if no objects of the class exist¹.
3. **Scope Resolution**: A static member function can also be accessed using the class name through the scope resolution operator¹.
4. **Limited Access**: A static member function can access static data members and static member functions inside or outside of the class¹. However, it cannot access non-static data members or call non-static member functions⁵.
5. **No Current Object Pointer**: Static member functions have a scope inside the class and cannot access the current object pointer¹.

Here's an example of how you might use a static function:

```cpp
#include <iostream>
using namespace std;

class Box {
private:
    static int length;
    static int breadth;
    static int height;

public:
    static void print() {
        cout << "The value of the length is: " << length << endl;
        cout << "The value of the breadth is: " << breadth << endl;
        cout << "The value of the height is: " << height << endl;
    }
};

int Box::length = 10;
int Box::breadth = 20;
int Box::height = 30;

int main() {
    Box b;
    cout << "Static member function is called through Object name: \\n" << endl;
    b.print();

    cout << "\\nStatic member function is called through Class name: \\n" << endl;
    Box::print();

    return 0;
}
```

In this example, the `print` function is a static member function of the `Box` class. It can be called either through an object of the class (`b.print()`) or directly through the class name (`Box::print()`)¹.

### Static funcion not inside class
In C++, a static function that is not inside a class is a function that has internal linkage¹. This means it is only accessible within the same translation unit (usually a .cpp file) that contains its definition¹. Here are some key points about static functions outside classes:

1. **Limited Visibility**: A static function remains visible only in file scope¹. This is a feature inherited from C¹. The recommended way to do it in C++ is using an anonymous namespace¹.

2. **Avoid Naming Conflicts**: The `static` keyword limits the visibility and linkage scope of the function to the current translation unit¹. That means that for a function, it can only be called from the current source file, and not from other source files¹. This helps avoid naming conflicts when the same function name is used in different source files¹.

Here's an example of how you might use a static function outside a class:

```cpp
// In a .cpp file
static void someRandomFunction() {
    // code
}

int main() {
    someRandomFunction();  // visible only within this file.
    return 0;
}
```

In this example, `someRandomFunction` is a static function that is only visible within the same file¹. If the function is actually called, you will get a linking error unless the function body is defined in the same file¹. The more pedantic technical term is actually not file but translation-unit since the body might be in an `#include`d header not in the actual file per-se¹.


