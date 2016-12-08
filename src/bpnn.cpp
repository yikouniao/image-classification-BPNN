#include "bpnn.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

namespace bpnn {

using std::cout;
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

BpNet::BpNet(double rate_h1_, double rate_o_, double err_thres_)
  : rate_h1(rate_h1_), rate_o(rate_o_), err_thres(err_thres_) {
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

void BpNet::Train(const std::string& filepath) {
  DataSet* samples = new DataSet;
  samples->GetTrainData(filepath);

  const size_t samples_num = samples->dataset.size();
  size_t train_times = 0;
  bool conv = false;
  while (!conv) {
    conv = true;
    ++train_times;
    cout << "The " << train_times << " times training...\n";
    for (size_t samples_order = 0; samples_order < samples_num; ++samples_order) {
      // Propagation
      array_h1 out_h1 = {0};
      GetOutH1(samples->dataset[samples_order].in, out_h1);

      array_o out_o = {0};
      GetOutO(out_h1, out_o);

      array_o err_o = {0};
      GetErrO(samples->dataset[samples_order].out, out_o, err_o);
      if (!CheckConv(err_o)) {
        conv = false;
      }
      if (!(samples_order % 5)) {
        cout << "Sample " << samples_order << " error: ";
        for (const auto& e_o : err_o) {
          cout << e_o << "\t";
        }
        cout << "\n";
      }

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
  }
  delete samples;
  cout << "\nTraining finished.\n\n";
}

void BpNet::Test(const std::string& filepath) {
  DataSet* testset = new DataSet;
  testset->GetTestData(filepath);
  auto it = testset->dataset.begin(), it_end = testset->dataset.end();

  cout << "\nTesting the model...\n";
  while (it != it_end) {    
    array_h1 out_h1 = {0};
    GetOutH1(it->in, out_h1);

    array_o out_o = {0};
    GetOutO(out_h1, it->out);

    array_o err_o = {0};
    GetErrO(it->out, out_o, err_o);

    cout << "The output error is:\n";
    for (const auto& e : err_o) {
      cout << e << "\t";
    }
    cout << "\n";
  }
  delete testset;
  cout << "\nTesting finished.\n\n";
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

} // namespace bpnn