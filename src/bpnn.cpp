#include "bpnn.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <opencv2/core.hpp>

namespace bpnn {

using std::cout;
using std::max_element;
using std::distance;
using data::DataSet;

namespace {

// get a random number in [-0.1 0.1]
inline double GetRand() {
  return 0.2 * rand() / RAND_MAX - 0.1;
}

// sigmod function
inline double Sigmod(double x) {
  return 1 / (1 + exp(-x));
}

} // namespace

BpNet::BpNet(double rate_h1_, double rate_o_, double err_thres_, double train_accu_rate_)
    : rate_h1(rate_h1_), rate_o(rate_o_), err_thres(err_thres_), train_accu_rate(train_accu_rate_) {
  srand(time(0)); // use current time as seed for random generator

// initialize w_h1
  for (auto& w_each_i : w_h1) {
    for (auto& w : w_each_i) {
      w = GetRand();
    }
  }

  // initialize w_o
  for (auto& w_each_h1 : w_o) {
    for (auto& w : w_each_h1) {
      w = GetRand();
    }
  }
}

BpNet::~BpNet() {}

void BpNet::Train(const std::string& img_filepath) {
  // get train data
  DataSet* samples = new DataSet;
  try {
    samples->GetTrainData(img_filepath);
  } catch (const std::exception& e) {
    std::cerr << "\nFailed to open file " << e.what() << "\n";
    return;
  }

  const clock_t begin_time = clock();
  const size_t samples_num = samples->dataset.size();
  size_t train_times = 0;
  bool conv = false;
  while (!conv) {
    ++train_times;
    cout << "The " << train_times << " times training...\n";
    size_t total = 0, correct = 0; // total and correct train samples number
    for (size_t samples_order = 0; samples_order < samples_num; ++samples_order) {
      // Propagation
      array_h1 out_h1 = {0};
      GetOutH1(samples->dataset[samples_order].in, out_h1);

      array_o out_o = {0};
      GetOutO(out_h1, out_o);

      array_o err_o = {0};
      GetErrO(samples->dataset[samples_order].out, out_o, err_o);
      if (CheckConv(err_o)) {
        ++correct;
      }
      ++total;

      // weights and thresholds update
      array_o sigma_o = {0};
      GetSigmaO(out_o, err_o, sigma_o);

      array_h1 err_h1 = {0};
      GetErrH1(sigma_o, err_h1);

      array_h1 sigma_h1 = {0};
      GetSigmaH1(out_h1, err_h1, sigma_h1);

      UpdateWO(out_h1, sigma_o);
      UpdateWH1(samples->dataset[samples_order].in, sigma_h1);
    }

    // reduce the learning rate while approaching the min cost
    UpdateRate(rate_h1);
    UpdateRate(rate_o);

    // check whether converges or not
    double correct_rate = double(correct) / total;
    cout << correct_rate * 100 << " per sent of train samples are correct.\n";
    if (correct_rate > train_accu_rate) {
      conv = true;
    }

  }
  std::cout << "\nTotal time spent: " << float(clock() - begin_time) / CLOCKS_PER_SEC << "s.\n\n";
  delete samples;
  cout << "\nTraining finished.\n\n";
}

void BpNet::Test(const std::string& img_filepath) {
  // get test data
  DataSet* testset = new DataSet;
  try {
    testset->GetTestData(img_filepath);
  } catch (const std::exception& e) {
    std::cerr << "\nFailed to open file " << e.what() << "\n";
    return;
  }
  auto it = testset->dataset.begin(), it_end = testset->dataset.end();

  cout << "\nTesting the model...\n";
  size_t total = 0, correct = 0; // total and correct test samples number
  while (it != it_end) {    
    array_h1 out_h1 = {0};
    GetOutH1(it->in, out_h1);

    array_o out_o = {0};
    GetOutO(out_h1, out_o);

    // In the test stage, select the class with biggest output value as the samples' class.
    // These two variables store the class number of groundtrue and test value.
    int c_groundtrue = distance(it->out.begin(), max_element(it->out.begin(), it->out.end()));
    int c_output = distance(out_o.begin(), max_element(out_o.begin(), out_o.end()));
    cout << "Groundtrue: " << c_groundtrue << " output: " << c_output;
    if (c_groundtrue == c_output) {
      cout << " TRUE\n";
      ++correct;
    } else {
      cout << " FALSE\n";
    }

    ++total;
    ++it;
  }
  delete testset;
  cout << "\nTesting finished.\nTotal accuracy: " << double(correct) / total << "\n";
}

void BpNet::FileWrite(const std::string& weight_file) {
  cout << "Writing weight into file...\n";
  cv::FileStorage fs(weight_file, cv::FileStorage::WRITE);

  // write w_h1
  {
    fs << "w_h1" << "[";
    auto it = w_h1.begin(), it_end = w_h1.end();
    while (it != it_end) {
      auto it_in = it->begin(), it_in_end = it->end();
      while (it_in != it_in_end) {
        fs << *it_in;
        ++it_in;
      }
      ++it;
    }
    fs << "]";
  }

  // write w_o
  {
    fs << "w_o" << "[";
    auto it = w_o.begin(), it_end = w_o.end();
    while (it != it_end) {
      auto it_in = it->begin(), it_in_end = it->end();
      while (it_in != it_in_end) {
        fs << *it_in;
        ++it_in;
      }
      ++it;
    }
    fs << "]";
  }

  // write rate_h1, rate_o and err_thres
  fs << "rate_h1" << rate_h1;
  fs << "rate_o" << rate_o;
  fs << "err_thres" << err_thres;

  fs.release();
  cout << "Write done.\n";
}

int BpNet::FileRead(const std::string& weight_file) {
  cout << "Reading weight from file...\n";
  cv::FileStorage fs;
  fs.open(weight_file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Failed to open " + weight_file << "\n";
    return -1;
  }

  // read w_h1
  {
    cv::FileNode n = fs["w_h1"];
    cv::FileNodeIterator fit = n.begin();
    auto it = w_h1.begin(), it_end = w_h1.end();
    while (it != it_end) {
      auto it_in = it->begin(), it_in_end = it->end();
      while (it_in != it_in_end) {
        *it_in = (double)*fit;
        ++it_in;
      }
      ++it;
    }
  }

  // read w_o
  {
    cv::FileNode n = fs["w_o"];
    cv::FileNodeIterator fit = n.begin();
    auto it = w_o.begin(), it_end = w_o.end();
    while (it != it_end) {
      auto it_in = it->begin(), it_in_end = it->end();
      while (it_in != it_in_end) {
        *it_in = (double)*fit;
        ++it_in;
      }
      ++it;
    }
  }

  // read rate_h1, rate_o and err_thres
  fs["rate_h1"] >> rate_h1;
  fs["rate_o"] >> rate_o;
  fs["err_thres"] >> err_thres;

  fs.release();
  cout << "Read done.\n";
  return 0;
}

void BpNet::GetOutH1(const array_i& in, array_h1& out_h1) {
  for (size_t i = 0; i < HIDEN1 - 1; ++i) {
    for (size_t j = 0; j < IN; ++j) {
      out_h1[i] += in[j] * w_h1[j][i];
    }
    out_h1[i] = Sigmod(out_h1[i]);
  }
  out_h1[HIDEN1 - 1] = 1;
}

void BpNet::GetOutO(const array_h1& out_h1, array_o& out_o) {
  for (size_t i = 0; i < OUT; ++i) {
    for (size_t j = 0; j < HIDEN1; ++j) {
      out_o[i] += out_h1[j] * w_o[j][i];
    }
    out_o[i] = Sigmod(out_o[i]);
  }
}

void BpNet::GetErrO(const array_o& out, const array_o& out_o, array_o& err_o) {
  for (size_t i = 0; i < OUT; ++i) {
    err_o[i] = out_o[i] - out[i];
  }
}

bool BpNet::CheckConv(const array_o& err_o) {
  for (const auto& e_o : err_o) {
    if (e_o < err_thres && e_o > -err_thres) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

void BpNet::GetSigmaO(const array_o& out_o, const array_o& err_o, array_o& sigma_o) {
  for (size_t i = 0; i < OUT; ++i) {
    sigma_o[i] = err_o[i] * out_o[i] * (1 - out_o[i]);
  }
}

void BpNet::GetErrH1(const array_o& sigma_o, array_h1& err_h1) {
  for (size_t i = 0; i < HIDEN1 - 1; ++i) {
    for (size_t j = 0; j < OUT; ++j) {
      err_h1[i] += sigma_o[j] * w_o[i][j];
    }
  }
}

void BpNet::GetSigmaH1(const array_h1& out_h1, const array_h1& err_h1, array_h1& sigma_h1) {
  for (size_t i = 0; i < HIDEN1 - 1; ++i) {
    sigma_h1[i] = err_h1[i] * out_h1[i] * (1 - out_h1[i]);
  }
}

void BpNet::UpdateWO(const array_h1& out_h1, const array_o& sigma_o) {
  for (size_t i = 0; i < OUT; ++i) {
    double delta_w = -rate_o * sigma_o[i];
    for (size_t j = 0; j < HIDEN1; ++j) {
      w_o[j][i] += delta_w * out_h1[j];
    }
  }
}

void BpNet::UpdateWH1(const array_i& in, const array_h1& sigma_h1) {
  for (size_t i = 0; i < HIDEN1 - 1; ++i) {
    double delta_w = -rate_h1 * sigma_h1[i];
    for (size_t j = 0; j < IN; ++j) {
      w_h1[j][i] += delta_w * in[j];
    }
  }
}

void BpNet::UpdateRate(double& rate) {
  // reduce the learning rate while approaching the min cost
  if (rate > .5) {
    rate -= .01;
  } else if (rate > .3) {
    rate -= .05;
  } else if (rate > .1) {
    rate -= .02;
  }
}

} // namespace bpnn