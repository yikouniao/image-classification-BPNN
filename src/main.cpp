/* Solve a XOR problem using BP neural net with one hiden layer. */

#include <iostream>
#include "bpnn.h"

using bpnn::BpNet;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Please input the file path of image data.\n";
    return -1;
  }
  BpNet* bp_net = new BpNet();
  bp_net->Train(argv[1]);
  bp_net->Test(argv[1]);
  delete bp_net;
  return 0;
}