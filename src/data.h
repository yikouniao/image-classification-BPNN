#pragma once

#include <array>
#include <vector>

// The number of actual input nodes shold be IN - 1. The special node with
// value 1 works the same as thresholds of next layer.
#define IN 3
#define OUT 1

using array_i = std::array<double, IN>;
using array_o = std::array<double, OUT>;

struct Data {
  array_i in;
  array_o out;
};

extern const std::vector<Data> samples;