#pragma once

/* This is a BP net model which has only one hiden layer.
* The numbers of input/hiden/output nodes are defined in sample.h
* Activation function: Sigmod function f(x) = 1 / (1 + e^(-x))
* Hiden layer:
*   layer 1
*     nodes number: 2
*     learning rate of weight value: fixed as 0.6
* Output layer:
*   learning rate of weight value: fixed as 0.6
* Convergence checking:
*   |error| < 0.008 */

#include "data.h"

// The number of actual hiden nodes shold be HIDEN1 - 1. The special node with
// value 1 works the same as thresholds of next layer.
#define HIDEN1 3

using array_h1 = std::array<double, HIDEN1>;
// When generating weights between hiden nodes and input nodes, leave out the
// last node in hiden layer because it's a constant 1.
using array_h1_w = std::array<double, HIDEN1 - 1>;

class BpNet {
public:
  BpNet(double rate_h1_ = 0.6, double rate_o_ = 0.6, double err_thres_ = 0.008);
  ~BpNet();

  // train the neural net
  void Train();

  // test the model
  void Test();

private:
  std::array<array_h1_w, IN> w_h1; // the input weights of hiden nodes in the 1st layer
  std::array<array_o, HIDEN1> w_o; // the input weights of output nodes
  double rate_h1; // learning rate of hiden nodes in the 1st layer
  double rate_o; // learning rate of output nodes
  double err_thres; // threshold for convergence checking

  // compute the output of hiden nodes in the 1st layer
  void GetOutH1(const array_i& in, array_h1& out_h1);

  // compute the output of output nodes
  void GetOutO(const array_h1& out_h1, array_o& out_o);

  // compute the errors of output nodes
  void GetErrO(const array_o& out, const array_o& out_o, array_o& err_o);

  // check weither the errors are acceptable
  bool CheckConv(const array_o& err_o);

  // compute the sigma for output nodes
  // sigma_o = err_o * samples.out * (1 - samples.out)
  void GetSigmaO(const array_o& out_o, const array_o& err_o, array_o& sigma_o);

  // compute the errors of hiden nodes in the 1st layer
  void GetErrH1(const array_o& sigma_o, array_h1& err_h1);

  // compute the sigma for hiden nodes in the 1st layer
  // sigma_h1 = err_h1 * out_h1 * (1 - out_h1)
  void GetSigmaH1(const array_h1& out_h1, const array_h1& err_h1, array_h1& sigma_h1);

  // update the input weights of output nodes
  void UpdateWO(const array_h1& out_h1, const array_o& sigma_o);

  // update the input weights of hiden nodes in the 1st layer
  void UpdateWH1(const array_i& in, const array_h1& sigma_h1);
};