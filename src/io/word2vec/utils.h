#ifndef WORD2VEC_UTILS_H
#define WORD2VEC_UTILS_H

#include <fstream>

namespace word2vec {

namespace utils {

  int64_t size(std::ifstream&);
  void seek(std::ifstream&, int64_t);
}

}

#endif
