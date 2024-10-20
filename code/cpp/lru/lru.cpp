




#include <cstdio>
#include <cassert>
#include <list>
#include <map>

class Cache {
public:
  Cache(int cap) {
    cap_ = cap;
  }
  int get(int key) {
    if(map_.find(key) != map_.end()) {
      int value = map_[key]->second;
      moveToFront(key, value);
      return value;
    }
    return -1;

  }

  void put(int key, int val) {
    if(map_.find(key) != map_.end()) {
      moveToFront(key, val);
      return;
    }
    if(items_.size() == cap_) {
      // remove the tail key from list and map
      int tail_key = items_.back().second;
      auto tail_key_iter = map_[tail_key]; 
      // does the order matter?
      map_.erase(tail_key);
      // items_.erase(tail_key_iter);
      items_.pop_back();
    }
    items_.push_front(std::make_pair(key, val));
    map_[key] = items_.begin();
  }

private:
  void moveToFront(int key, int val) {
    items_.erase(map_[key]);
    items_.push_front(std::make_pair(key, val));
    // map_.insert(std::make_pair(key, items_.begin()));
    map_[key] = items_.begin();
  }
  int cap_;
  std::list<std::pair<int, int>> items_;
  std::map<int, std::list<std::pair<int, int>>::iterator> map_;
};
int main() {
  auto cache =   Cache(3);
  cache.put(1, 1 );
  cache.put(2, 2 );
  cache.put(3, 3);
  assert(cache.get(1) == 1);
  assert(cache.get(2) == 2);
  assert(cache.get(3) == 3);

  cache.put(4, 4);
  assert(cache.get(1) == -1);

  assert(cache.get(2) == 2);
  cache.put(5, 5);
  assert(cache.get(3) == -1);
  printf("OK\n");

}
