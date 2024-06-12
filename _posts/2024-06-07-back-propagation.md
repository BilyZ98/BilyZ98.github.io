---
layout: post
title: Backpropogation C++ Implementation
date: 2024-06-06 07:59:00-0400
description: Artificial Intelligence
tags: ai ml 
categories: ai
featured: false
---

## Backpropagation code example 
[Deep learning models from scratch using C++ and Python](https://alexgl-github.github.io/github/jekyll/2021/04/16/Dense_layer.html)
```cpp
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <functional>
#include <array>
#include <iterator>


using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

static auto ones_initializer = []() -> float {
  return 1.0;
};

template<size_t num_inputs, size_t num_outputs, typename T = float, typename Initializer = decltype(ones_initializer)>
class Network {
public:
  Network(Initializer initializer = ones_initializer) {
    for (size_t i = 0; i < num_outputs; ++i) {
      for (size_t j = 0; j < num_inputs; ++j) {
        weights_[i][j] = initializer();
      }
    }
  }

  std::array<T, num_outputs> Forward(const std::array<T, num_inputs>& inputs) {
    std::array<T, num_outputs> outputs;
    for (size_t i = 0; i < num_outputs; ++i) {
      outputs[i] = std::inner_product(std::begin(inputs), std::end(inputs), std::begin(weights_[i]), T{0});
    }
    return outputs;
  }

  void PrintWeights() {
    for (size_t i = 0; i < num_outputs; ++i) {
      for (size_t j = 0; j < num_inputs; ++j) {
        std::cout << weights_[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }


  void Backward(const std::array<T, num_inputs>& inputs, const std::array<T, num_outputs>& dloss_dy, T learning_rate) {
    for (size_t i = 0; i < num_outputs; ++i) {
      for (size_t j = 0; j < num_inputs; ++j) {
        weights_[i][j] -= learning_rate * dloss_dy[i] * inputs[j];
      }
    }
  
  }
private:
  std::array<std::array<T, num_inputs>, num_outputs> weights_;
};

template<size_t num_inputs, typename T = float>
class MSE {
public:
  T operator()(const std::array<T, num_inputs>& outputs, const std::array<T, num_inputs>& targets) {
    T sum = 0;
    for (size_t i = 0; i < num_inputs; ++i) {
      sum += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
    }
    return sum / num_inputs;
  }

  std::array<T, num_inputs> Gradient(const std::array<T, num_inputs>& outputs, const std::array<T, num_inputs>& targets) {
    std::array<T, num_inputs> gradient;
    for (size_t i = 0; i < num_inputs; ++i) {
      gradient[i] = 2 * (outputs[i] - targets[i]);
    }
    return gradient;
  } 
};


int main() {
  Network<3, 2> network;
  const int num_iterators = 10;
  network.PrintWeights();

  std::array<float, 3> inputs = {2.0, 0.5, 1.0};
  std::array<float, 2> targets = {1.5, 1.0};
  MSE<2> mse;

  for(size_t i=0; i < num_iterators; i++ ) {
    auto start = high_resolution_clock::now();
    auto yhat = network.Forward(inputs);
    std::cout << "Loss: " << mse(yhat, targets) << std::endl;
    std::cout << "Gradient: " << mse.Gradient(yhat, targets)[0] << std::endl;
    std::cout << "yhat:" <<  yhat[0] << " " << yhat[1] << std::endl;
    auto loss = mse(yhat, targets);
    auto dloss_dy = mse.Gradient(yhat, targets);
    network.Backward(inputs, dloss_dy, 0.1);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;
    yhat = network.Forward(inputs); 
    std::cout << "yhat" << yhat[0] << " " << yhat[1] << std::endl;
    std::cout << "----------------" << std::endl;


  }

  // for (int i = 0; i < 1000; ++i) {
  //   auto outputs = network.Forward(inputs);
  //   network.Backward(inputs, outputs, targets, 0.1);
  // }

  network.PrintWeights();
  return 0;
}
```




