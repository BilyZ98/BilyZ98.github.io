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
