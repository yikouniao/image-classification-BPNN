/* Solve a XOR problem using BP neural net with one hiden layer. */

#include <iostream>
#include "bpnn.h"

using namespace std;

int main(int argc, char** argv) {
  BpNet bp_net;
  bp_net.Train();
  bp_net.Test();
}