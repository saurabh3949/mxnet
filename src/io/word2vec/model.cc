#include "model.h"

#include <iostream>
#include <assert.h>
#include <algorithm>

namespace word2vec {

Model::Model(std::shared_ptr<Args> args,
             int32_t seed)
  : rng(seed)
{
  args_ = args;
  negpos = 0;
  nexamples_ = 1;
}

Model::~Model() {
}

void Model::setTargetCounts(const std::vector<int64_t>& counts) {
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives.push_back(i);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

void Model::update(int32_t center, int32_t target) {
    for (auto i = 0; i < args_->neg; ++i) {
        int32_t neg = getNegative(target);
    }
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

}
