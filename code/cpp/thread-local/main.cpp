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
