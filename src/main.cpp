/* Solve a XOR problem using BP neural net with one hiden layer. */

#include <iostream>
#include <string>
#include "bpnn.h"

using bpnn::BpNet;

// argv[1] : filepath of image dataset
// argv[2] : 1 corresponds to use existing weight data
//           0 corresponds to re-train the model
// argv[3] : file name and path of weight recoder
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Please input the file path of image data.\n";
    return -1;
  } else if (argc < 3) {
    std::cerr << "Please verify whether to use existing weight data or not, "
              << "and input the file of weight data.\n";
    return -1;
  }

  BpNet* bp_net = new BpNet();
  int flag = 0;
  if (std::stoi(argv[2])) {
    flag = bp_net->FileRead(argv[3]);
  }
  if (flag || !std::stoi(argv[2])) {
    bp_net->Train(argv[1]);
    bp_net->FileWrite(argv[3]);
  }

  bp_net->Test(argv[1]);
  delete bp_net;

  return 0;
}