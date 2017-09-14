/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file iter_word2vec_2.cc
 * \brief Word2Vec data iterator
 */

#include <mxnet/io.h>
#include <dmlc/parameter.h>
#include <dmlc/threadediter.h>
#include <dmlc/input_split_shuffle.h>
#include <dmlc/recordio.h>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/omp.h>
#include <dmlc/common.h>
#include <dmlc/timer.h>
#include <type_traits>
#include "./image_iter_common.h"
#include "./inst_vector.h"
#include "../common/utils.h"

#include "./word2vec/args.h"
#include "./word2vec/dictionary.h"
#include "./word2vec/model.h"
#include "./word2vec/real.h"
#include "./word2vec/utils.h"

namespace mxnet {
namespace io {

/*!
 * \brief a list of (center word, target words) pairs
 */
template<typename DType = real_t>
class InstVectorWord2Vec {
 public:
  /*! \brief return the number of center word, target words) pairs */
  inline size_t Size(void) const {
    return index_.size();
  }
  // get index
  inline unsigned Index(unsigned i) const {
    return index_[i];
  }
  // instance
  /* \brief get the i-th (center word, target words) pair */
  inline DataInst operator[](size_t i) const {
    DataInst inst;
    inst.index = index_[i];
    inst.data.push_back(TBlob(center_[i]));
    inst.data.push_back(TBlob(random_[i]));
    return inst;
  }
  /* \brief get the last (center word, target words) pairs */
  inline DataInst Back() const {
    return (*this)[Size() - 1];
  }
  inline void Clear(void) {
    index_.clear();
    center_.Clear();
    random_.Clear();
  }
  /*
   * \brief push a (center word, target words) pair
   * only reserved the space, while the data is not copied
   */
  inline void Push(unsigned index,
                   mshadow::Shape<1> center_shape,
                   mshadow::Shape<1> random_shape) {
    index_.push_back(index);
    center_.Push(center_shape);
    random_.Push(random_shape);
  }
  /*! \return the data content */
  inline const TensorVector<1, DType>& center() const {
    return center_;
  }

  inline const TensorVector<1, DType>& random() const {
  return random_;
}

 private:
  /*! \brief index of the data */
  std::vector<unsigned> index_;
  // center word
  TensorVector<1, DType> center_;
  // target words (centext + k negative samples)
  TensorVector<1, DType> random_;

};


struct Word2VecParam : public dmlc::Parameter<Word2VecParam> {
  /*! \brief path of input corpus */
  std::string file_path;
  /*! \brief path of vocabulary to be saved */
  std::string vocab_path;
  /*! \brief mode of algorithm - 'skipgram' or 'cbow' */
  std::string mode;
  /*! \brief batch size */
  int batch_size;
  /*! \brief minimal number of word occurences */
  int min_count;
  /*! \brief number of buckets */
  int num_buckets;
  /*! \brief sampling threshold */
  float sampling_threshold;
  /*! \brief size of the context window */
  int window_size;
  /*! \brief negative samples */
  int negative_samples;
  /*! \brief number of threads */
  int preprocess_threads;
  // declare parameters
  DMLC_DECLARE_PARAMETER(Word2VecParam) {
    DMLC_DECLARE_FIELD(file_path)
        .describe("The input text corpus.");
    DMLC_DECLARE_FIELD(vocab_path).set_default("./word2vec_vocab")
        .describe("The path of vocabulary to be saved.");
    DMLC_DECLARE_FIELD(mode).set_default("skipgram")
        .describe("Algorithm Param: 'skpgram' or 'cbow'.");
    DMLC_DECLARE_FIELD(batch_size).set_lower_bound(1).set_default(256)
        .describe("Batch Param: Batch Size.");
    DMLC_DECLARE_FIELD(min_count).set_lower_bound(1).set_default(5)
        .describe("minimal number of word occurences.");
    DMLC_DECLARE_FIELD(num_buckets).set_lower_bound(1000000).set_default(2000000)
        .describe("number of buckets.");
    DMLC_DECLARE_FIELD(sampling_threshold).set_default(0.0001f)
        .describe("sampling threshold.");
    DMLC_DECLARE_FIELD(window_size).set_lower_bound(1).set_default(5)
        .describe("The size of the context window.");
    DMLC_DECLARE_FIELD(negative_samples).set_lower_bound(1).set_default(5)
        .describe("The number of negative samples.");
    DMLC_DECLARE_FIELD(preprocess_threads).set_lower_bound(1).set_default(4)
        .describe("The number of threads to do preprocessing.");
  }
};

// parser to parse corpus for word pairs
template<typename DType>
class Word2VecParser {
 public:
  // initialize the parser
  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs);

  // seek all workers at their respective positions.
  inline void BeforeFirst(void) {
    overflow = false;
    tokens_processed = false;
    n_parsed_ = 0;
    int threadget = word2vec_param_.preprocess_threads;
    for (int32_t i = 0; i < threadget; ++i) {
      std::ifstream &ifs = istreams_.at(i);
      word2vec::utils::seek(ifs, i * input_size / threadget);
    }
  }
  // parse next set of records, return an array of
  // instance vector to the user
  inline bool ParseNext(DataBatch *out);

 private:
  /*! \brief Read and process next couple of sentences */
  inline void ParseChunk();

  /*! \brief Parameters */
  Word2VecParam word2vec_param_;
  BatchParam batch_param_;
  PrefetcherParam prefetch_param_;

  /*! \brief Word2Vec compute related classes */
  std::shared_ptr<word2vec::Args> args_;
  std::shared_ptr<word2vec::Dictionary> dict_;
  /*! \brief workers for preprocessing */
  std::vector<word2vec::Model> workers_;
  /*! \brief input streams for each worker*/
  std::vector<std::ifstream> istreams_;
  int64_t tokens_processed;
  int64_t tokens_total;
  int64_t input_size;

  static const int kRandMagic = 111;
  common::RANDOM_ENGINE rnd_;


  /*! \brief temporary results */
  std::vector<InstVectorWord2Vec<DType>> temp_;
  /*! \brief internal instance order */
  std::vector<std::pair<unsigned, unsigned> > inst_order_;
  unsigned inst_index_;
  /*! \brief internal counter tracking number of already parsed entries */
  unsigned n_parsed_;
  /*! \brief overflow marker */
  bool overflow;
  /*! \brief unit size */
  std::vector<size_t> unit_size_;
};

template<typename DType>
inline void Word2VecParser<DType>::Init(
    const std::vector<std::pair<std::string, std::string> >& kwargs) {
  // initialize parameters
  batch_param_.InitAllowUnknown(kwargs);
  prefetch_param_.InitAllowUnknown(kwargs);
  word2vec_param_.InitAllowUnknown(kwargs);

  // Initialize Word2Vec params
  args_ = std::make_shared<word2vec::Args>();
  args_->input = word2vec_param_.file_path;

  dict_ = std::make_shared<word2vec::Dictionary>(args_);
  std::ifstream ifs(args_->input);
  CHECK(ifs.is_open()) << "Input file cannot be opened!";
  input_size = word2vec::utils::size(ifs);
  ifs.seekg(std::streampos(0));
  dict_->readFromFile(ifs);
  ifs.close();
  tokens_processed = 0;
  tokens_total = dict_->ntokens();

  std::ofstream ofs(word2vec_param_.vocab_path);
  CHECK(ofs.is_open()) << "Cannot open output file to write vocabulary!";
  dict_->save_words(ofs);
  ofs.close();
  LOG(INFO) << "Saved vocabulary to " << word2vec_param_.vocab_path;

  n_parsed_ = 0;
  overflow = false;
  rnd_.seed(kRandMagic + 1);

  int maxthread, threadget;
  #pragma omp parallel
  {
    // TODO: Be conservative, set number of real cores
    maxthread = std::max(omp_get_num_procs() - 1, 1);
  }
  word2vec_param_.preprocess_threads = std::min(maxthread, word2vec_param_.preprocess_threads);
  #pragma omp parallel num_threads(word2vec_param_.preprocess_threads)
  {
    threadget = omp_get_num_threads();
  }
  word2vec_param_.preprocess_threads = threadget;

  for (int32_t i = 0; i < threadget; ++i) {
    workers_.emplace_back(args_, i);
    workers_.back().setTargetCounts(dict_->getCounts());
    istreams_.emplace_back(args_->input);
    word2vec::utils::seek(istreams_.back(), i * input_size / threadget);
  }

  LOG(INFO) << "Word2VecParser initialized successfully! " 
            << "Using " << threadget << " threads for preprocessing..";
}

template<typename DType>
inline bool Word2VecParser<DType>::ParseNext(DataBatch *out) {
  if (overflow)
    return false;
  unsigned current_size = 0;
  out->index.resize(batch_param_.batch_size);
  while (current_size < batch_param_.batch_size) {
    int n_to_copy;
    if (n_parsed_ == 0) {
      if (tokens_processed < tokens_total) {
        inst_order_.clear();
        inst_index_ = 0;
        ParseChunk();
        unsigned n_read = 0;
        for (unsigned i = 0; i < temp_.size(); ++i) {
          const InstVectorWord2Vec<DType>& tmp = temp_[i];
          for (unsigned j = 0; j < tmp.Size(); ++j) {
            inst_order_.push_back(std::make_pair(i, j));
          }
          n_read += tmp.Size();
        }
        n_to_copy = std::min(n_read, batch_param_.batch_size - current_size);
        n_parsed_ = n_read - n_to_copy;
        
        // shuffle instance order if needed. TODO: take a flag from user
        std::shuffle(inst_order_.begin(), inst_order_.end(), rnd_);
        
      } else {
        if (current_size == 0) return false;
        CHECK(!overflow) << "number of input word pairs must be bigger than the batch size";
        if (batch_param_.round_batch != 0) {
          overflow = true;
          tokens_processed = 0;
        } else {
          current_size = batch_param_.batch_size;
        }
        out->num_batch_padd = batch_param_.batch_size - current_size;
        n_to_copy = 0;
      }
    } else {
      n_to_copy = std::min(n_parsed_, batch_param_.batch_size - current_size);
      n_parsed_ -= n_to_copy;
    }

    // InitBatch
    if (out->data.size() == 0 && n_to_copy != 0) {
      std::pair<unsigned, unsigned> place = inst_order_[inst_index_];
      const DataInst& first_batch = temp_[place.first][place.second];
      out->data.resize(first_batch.data.size());
      unit_size_.resize(first_batch.data.size());
      for (size_t i = 0; i < out->data.size(); ++i) {
        TShape src_shape = first_batch.data[i].shape_;
        int src_type_flag = first_batch.data[i].type_flag_;
        // init object attributes
        std::vector<index_t> shape_vec;
        shape_vec.push_back(batch_param_.batch_size);
        for (index_t dim = 0; dim < src_shape.ndim(); ++dim) {
          shape_vec.push_back(src_shape[dim]);
        }
        TShape dst_shape(shape_vec.begin(), shape_vec.end());
        auto dtype = prefetch_param_.dtype
          ? prefetch_param_.dtype.value()
          : first_batch.data[i].type_flag_;
        out->data.at(i) = NDArray(dst_shape, Context::CPUPinned(0), false, src_type_flag);
        unit_size_[i] = src_shape.Size();
      }
    }

    // Copy
    #pragma omp parallel for num_threads(word2vec_param_.preprocess_threads)
    for (int i = 0; i < n_to_copy; ++i) {
      std::pair<unsigned, unsigned> place = inst_order_[inst_index_ + i];
      const DataInst& batch = temp_[place.first][place.second];
      for (unsigned j = 0; j < batch.data.size(); ++j) {
        CHECK_EQ(unit_size_[j], batch.data[j].Size());
        MSHADOW_TYPE_SWITCH(out->data[j].data().type_flag_, dtype, {
        mshadow::Copy(
            out->data[j].data().FlatTo1D<cpu, dtype>().Slice((current_size + i) * unit_size_[j],
              (current_size + i + 1) * unit_size_[j]),
            batch.data[j].get_with_shape<cpu, 1, dtype>(mshadow::Shape1(unit_size_[j])));
        });
      }
    }
    inst_index_ += n_to_copy;
    current_size += n_to_copy;
  }
  return true;
}

template<typename DType>
inline void Word2VecParser<DType>::ParseChunk() {
  temp_.resize(word2vec_param_.preprocess_threads);
  #pragma omp parallel num_threads(word2vec_param_.preprocess_threads)
  {
    CHECK(omp_get_num_threads() == word2vec_param_.preprocess_threads);
    int tid = omp_get_thread_num();

    word2vec::Model &model = workers_[tid];
    std::ifstream &ifs = istreams_.at(tid);
    InstVectorWord2Vec<DType> &out = temp_[tid];
    out.Clear();

    int64_t localTokenCount = 0;
    std::vector<int32_t> line, labels;

    // Process next 500 sentences. TODO: take this as a parameter from the user.
    for (int total=0; total < 500; total++) {
      if (tokens_processed < tokens_total){
        localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
        std::uniform_int_distribution<> uniform(1, word2vec_param_.window_size);
        for (int32_t w = 0; w < line.size(); w++) {
          int32_t boundary = uniform(model.rng);
          for (int32_t c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line.size()) {
              out.Push(0, mshadow::Shape1(1), mshadow::Shape1(word2vec_param_.negative_samples+1));
              mshadow::Tensor<cpu, 1, DType> center = out.center().Back();
              mshadow::Tensor<cpu, 1, DType> random = out.random().Back();
              center[0] = line[w];
              random[0] = line[w + c];
              for (int i = 1; i <= word2vec_param_.negative_samples; i++){
                random[i] = model.getNegative(line[w + c]);
              }
            }
          }
        }
        #pragma omp atomic
        tokens_processed += localTokenCount;
        localTokenCount = 0;
      } else {
        break;
      }
    }
  }
}


template<typename DType = real_t>
class Word2VecIter : public IIterator<DataBatch> {
 public:
    Word2VecIter() : out_(nullptr) { }

    virtual ~Word2VecIter(void) {
      iter_.Destroy();
    }

    virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
      prefetch_param_.InitAllowUnknown(kwargs);
      parser_.Init(kwargs);
      // maximum prefetch threaded iter internal size
      const int kMaxPrefetchBuffer = 1024;
      // init thread iter
      iter_.set_max_capacity(kMaxPrefetchBuffer);
      // init thread iter
      iter_.Init([this](DataBatch **dptr) {
          if (*dptr == nullptr) {
            *dptr = new DataBatch();
          }
          return parser_.ParseNext(*dptr);
          },
          [this]() { parser_.BeforeFirst(); });
    }

    virtual void BeforeFirst(void) {
      iter_.BeforeFirst();
    }

    // From iter_prefetcher.h
    virtual bool Next(void) {
      if (out_ != nullptr) {
        recycle_queue_.push(out_); out_ = nullptr;
      }
      // do recycle
      if (recycle_queue_.size() == prefetch_param_.prefetch_buffer) {
        DataBatch *old_batch =  recycle_queue_.front();
        // can be more efficient on engine
        for (NDArray& arr : old_batch->data) {
          arr.WaitToWrite();
        }
        recycle_queue_.pop();
        iter_.Recycle(&old_batch);
      }
      return iter_.Next(&out_);
    }

    virtual const DataBatch &Value(void) const {
      return *out_;
    }

 private:
    /*! \brief Backend thread */
    dmlc::ThreadedIter<DataBatch> iter_;
    /*! \brief Parameters */
    PrefetcherParam prefetch_param_;
    /*! \brief output data */
    DataBatch *out_;
    /*! \brief queue to be recycled */
    std::queue<DataBatch*> recycle_queue_;
    /* \brief parser */
    Word2VecParser<DType> parser_;
};

DMLC_REGISTER_PARAMETER(Word2VecParam);

MXNET_REGISTER_IO_ITER(Word2VecIter)
.describe(R"code(Iterates on word pairs for Word2Vec.

Data contains an NDArray of (batch_size,1) with center words' indices.
Label contains an NDArray of (batch_size,1+k) with the target words indices and
k negative samples for each instance.

)code" ADD_FILELINE)
.add_arguments(Word2VecParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new Word2VecIter<int32_t>();
    });

}  // namespace io
}  // namespace mxnet
