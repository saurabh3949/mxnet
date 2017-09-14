#ifndef WORD2VEC_ARGS_H
#define WORD2VEC_ARGS_H

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace word2vec {

enum class model_name : int {cbow=1, sg, sup};
enum class loss_name : int {hs=1, ns, softmax};

class Args {
  public:
    Args();
    std::string input;
    int ws;
    int minCount;
    int minCountLabel;
    int neg;
    int wordNgrams;
    loss_name loss;
    int bucket;
    int maxn;
    int minn;
    double t;
    int verbose;
};

}

#endif
