#include "args.h"

#include <stdlib.h>

#include <iostream>

namespace word2vec {

Args::Args() {
  ws = 5;
  minCount = 5;
  minCountLabel = 0;
  neg = 5;
  wordNgrams = 1;
  loss = loss_name::ns;
  bucket = 2000000;
  maxn = 0;
  minn = 0;
  t = 1e-4;
  verbose = 2;
}

}
