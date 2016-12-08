#pragma once

#include <array>
#include <vector>

namespace data {

// The number of actual input nodes shold be IN - 1. The special node with
// value 1 works the same as thresholds of next layer.
#define IN (900 + 1)
#define OUT 70

using array_i = std::array<double, IN>;
using array_o = std::array<double, OUT>;

struct Data {
  array_i in;
  array_o out;
};

class DataSet {
 public:
  DataSet();
  ~DataSet();

  // Read train/test data from image files. filepath is the path of the dataset.
  void GetTrainData(const std::string& filepath);
  void GetTestData(const std::string& filepath);

  std::vector<Data> dataset;
};

} // namespace data