/* Solve a XOR problem using BP neural net with one hiden layer. */

#include <iostream>
#include "bpnn.h"

using bpnn::BpNet;

// argv[1] : filepath of image dataset
// argv[2] : true corresponds to use existing weight data
//           false corresponds to re-train the model
// argv[3] : file name and path of weight recoder
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Please input the file path of image data.\n";
    return -1;
  } else if (argc < 4) {
    std::cerr << "Please verify whether to use existing weight data or not, "
              << "and input the file of weight data.\n";
    return -1;
  }

  BpNet* bp_net = new BpNet();
  int flag = 0;
  if (argv[2] == "true") {
    flag = bp_net->FileRead(argv[3]);
  }
  if (flag || argv[2] == "false") {
    bp_net->Train(argv[1]);
    bp_net->FileWrite(argv[3]);
  }

  bp_net->Test(argv[1]);
  delete bp_net;

  return 0;
}