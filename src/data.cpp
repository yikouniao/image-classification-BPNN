#include "data.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace data {

using std::vector;
using std::string;
using cv::Mat;

namespace {

// There're 70 kinds of shape images as follow. Each type of shapes contains 20
// images, we choose 18 of which to train the network and the other 2 to test
// it. All images will be compressed into 30 * 30 pixels.
const vector<string> filename = {
  "apple", "bat", "beetle", "bell", "bird", "Bone", "bottle", "brick",
  "butterfly", "camel", "car", "carriage", "cattle", "cellular_phone",
  "chicken", "children", "chopper", "classic", "Comma", "crown", "cup", "deer",
  "device0", "device1", "device2", "device3", "device4", "device5", "device6",
  "device7", "device8", "device9", "dog", "elephant", "face", "fish",
  "flatfish", "fly", "fork", "fountain", "frog", "Glas", "guitar", "hammer",
  "hat", "HCircle", "Heart", "horse", "horseshoe", "jar", "key", "lizzard",
  "lmfish", "Misk", "octopus", "pencil", "personal_car", "pocket", "rat",
  "ray", "sea_snake", "shoe", "spoon", "spring", "stef", "teddy", "tree",
  "truck", "turtle", "watch"
};

const size_t img_size = 30; // img_size * img_size = IN

// Read dat from a img. ccnt is the counter of clusters.
void ReadDataFromImg(Mat& img, Data& dat, size_t ccnt) {
  auto it_dat = dat.in.begin();
  cv::resize(img, img, cv::Size(img_size, img_size));
  cv::MatConstIterator_<uchar> it_img = img.begin<uchar>(), it_img_end = img.end<uchar>();
  while (it_img != it_img_end) {
    *it_dat++ = *it_img++ > 125 ? 1 : 0;
  }
  *it_dat = 1; // the last input node for bias

  // output info
  dat.out.fill(0);
  dat.out[ccnt] = 1;
}

} // namespace

DataSet::DataSet() {}

DataSet::~DataSet() {}

void DataSet::GetTrainData(const string& filepath) {
  for (size_t ccnt = 0; ccnt < IN; ++ccnt) {
    int train_cnt = 18;
    while (train_cnt) {
      Data* dat = new Data;
      string fname = filepath + filename[ccnt] + "-" + std::to_string(train_cnt) + ".jpg";
      Mat img = cv::imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
      ReadDataFromImg(img, *dat, ccnt);
      dataset.insert(dataset.begin(), *dat);
      delete dat;
      --train_cnt;
    }
  }
}

void DataSet::GetTestData(const string& filepath) {
  for (size_t ccnt = 0; ccnt < IN; ++ccnt) {
    int test_cnt = 20;
    while (test_cnt > 18) {
      Data* dat = new Data;
      string fname = filepath + filename[ccnt] + "-" + std::to_string(test_cnt) + ".jpg";
      Mat img = cv::imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
      ReadDataFromImg(img, *dat, ccnt);
      dataset.insert(dataset.begin(), *dat);
      delete dat;
      --test_cnt;
    }
  }
}

} // namespace data