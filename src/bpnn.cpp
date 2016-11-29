#include "bpnn.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

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

BpNet::BpNet(double rate_w_h1_, double rate_w_o_, double rate_thres_h1_,
  double rate_thres_o_, double err_thres_)
  : rate_w_h1(rate_w_h1_), rate_w_o(rate_w_o_), rate_thres_h1(rate_thres_h1_),
  rate_thres_o(rate_thres_o_), err_thres(err_thres_) {
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

  // initialize thres_h1
  for (auto& thres_each_h1 : thres_h1) {
    thres_each_h1 = GetRand();
  }

  // initialize thres_o
  for (auto& thres_each_o : thres_o) {
    thres_each_o = GetRand();
  }
}

BpNet::~BpNet() {}

void BpNet::Train() {
  const size_t samples_num = samples.size();
  size_t train_times = 0;
  bool conv = false;
  while (!conv) {
    conv = true;
    ++train_times;
    cout << "The " << train_times << " times training...\n";
    for (size_t samples_order = 0; samples_order < samples_num; ++samples_order) {
      // Propagation
      array_h1 out_h1 = { 0 };
      GetOutH1(samples[samples_order].in, out_h1);

      array_o out_o = { 0 };
      GetOutO(out_h1, out_o);

      array_o err_o = { 0 };
      GetErrO(samples[samples_order].out, out_o, err_o);
      if (!CheckConv(err_o))
        conv = false;
      cout << "Sample " << samples_order << " error: ";
      for (const auto& e_o : err_o) {
        cout << e_o << "\t";
      }
      cout << "\n";

      // weights and thresholds update
      array_o sigma_o = { 0 };
      GetSigmaO(out_o, err_o, sigma_o);

      array_h1 err_h1 = { 0 };
      GetErrH1(sigma_o, err_h1);

      array_h1 sigma_h1 = { 0 };
      GetSigmaH1(out_h1, err_h1, sigma_h1);

      UpdateThresO(sigma_o);
      UpdateThresH1(sigma_h1);
      UpdateWO(out_h1, sigma_o);
      UpdateWH1(samples[samples_order].in, sigma_h1);
    }
  }
  cout << "\nTraining finished.\n\n";
}

void BpNet::Test() {
  while (1) {
    cout << "Test the model? (y/n):";
    char c;
    cin.get(c);
    if (c == 'y' || c == 'Y') {} else if (c == 'n' || c == 'N') {
      return;
    } else {
      continue;
    }

    cout << "Please input the data:\n";
    Data test = { { 0 },{ 0 } };
    for (auto& e : test.in) {
      cin >> e;
    }

    array_h1 out_h1 = { 0 };
    GetOutH1(test.in, out_h1);

    GetOutO(out_h1, test.out);

    cout << "\nThe output is:\n";
    for (const auto& e : test.out) {
      cout << e << "\t";
    }
    cout << "\n";
  }
}

void BpNet::GetOutH1(const array_i& in, array_h1& out_h1) {
  for (size_t i = 0; i < HIDEN1; ++i) {
    for (size_t j = 0; j < IN; ++j) {
      out_h1[i] += in[j] * w_h1[j][i];
    }
    out_h1[i] += thres_h1[i];
    out_h1[i] = Sigmod(out_h1[i]);
  }
}

void BpNet::GetOutO(const array_h1& out_h1, array_o& out_o) {
  for (size_t i = 0; i < OUT; ++i) {
    for (size_t j = 0; j < HIDEN1; ++j) {
      out_o[i] += out_h1[j] * w_o[j][i];
    }
    out_o[i] += thres_o[i];
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
  for (size_t i = 0; i < HIDEN1; ++i) {
    for (size_t j = 0; j < OUT; ++j) {
      err_h1[i] += sigma_o[j] * w_o[i][j];
    }
  }
}

void BpNet::GetSigmaH1(const array_h1& out_h1, const array_h1& err_h1, array_h1& sigma_h1) {
  for (size_t i = 0; i < HIDEN1; ++i) {
    sigma_h1[i] = err_h1[i] * out_h1[i] * (1 - out_h1[i]);
  }
}

void BpNet::UpdateThresO(const array_o& sigma_o) {
  for (size_t i = 0; i < OUT; ++i) {
    double delta_w = -rate_thres_o * sigma_o[i];
    thres_o[i] += delta_w;
  }
}

void BpNet::UpdateThresH1(const array_h1& sigma_h1) {
  for (size_t i = 0; i < HIDEN1; ++i) {
    double delta_w = -rate_thres_h1 * sigma_h1[i];
    thres_h1[i] += delta_w;
  }
}

void BpNet::UpdateWO(const array_h1& out_h1, const array_o& sigma_o) {
  for (size_t i = 0; i < OUT; ++i) {
    double delta_w = -rate_w_o * sigma_o[i];
    for (size_t j = 0; j < HIDEN1; ++j) {
      w_o[j][i] += delta_w * out_h1[j];
    }
  }
}

void BpNet::UpdateWH1(const array_i& in, const array_h1& sigma_h1) {
  for (size_t i = 0; i < HIDEN1; ++i) {
    double delta_w = -rate_w_h1 * sigma_h1[i];
    for (size_t j = 0; j < IN; ++j) {
      w_h1[j][i] += delta_w * in[j];
    }
  }
}