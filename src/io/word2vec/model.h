#ifndef WORD2VEC_MODEL_H
#define WORD2VEC_MODEL_H

#include <vector>
#include <random>
#include <utility>
#include <memory>

#include "args.h"
#include "real.h"

namespace word2vec {

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t count;
  bool binary;
};

class Model {
  private:
    std::shared_ptr<Args> args_;
    int32_t osz_;
    int64_t nexamples_;
    // used for negative sampling:
    std::vector<int32_t> negatives;
    size_t negpos;

    static bool comparePairs(const std::pair<real, int32_t>&,
                             const std::pair<real, int32_t>&);

    static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

  public:
    Model(std::shared_ptr<Args>, int32_t);
    ~Model();
    void update(int32_t center, int32_t target);
    void setTargetCounts(const std::vector<int64_t>&);
    void initTableNegatives(const std::vector<int64_t>&);
    int32_t getNegative(int32_t target);

    std::minstd_rand rng;
};

}

#endif
