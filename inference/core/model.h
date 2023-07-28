#pragma once

#include <condition_variable>
#include <map>

#include "benchmark/common/double_buffer_reloader.h"
#include "benchmark/proto/bench_conf.pb.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
namespace benchmark {

class Model {
 public:
  Model(const std::string& name, int predictor_num) {
    name_ = name;
    predictor_num_ = predictor_num;
    batchsize_internal_ = 100;
  }
  ~Model() {
    for (auto iter : sessions_) {
      auto sess = iter.second;
      sess->Close();
      delete sess;
    }
  }

  bool LoadGraph(const std::string& frozen_graph);
  bool InitSession(const std::string& conf_proto);
  bool ParseRunOptions(const std::string& run_options);
  bool ParseSamples(const std::string& sample_file);
  bool Warmup();

  /* Optim functions */
  void AddDeepCopyNode();
  int GetNumOfVirtualGPU(ConfigProto* config);
  void PlaceGraphToDevice(int gpu_idx);
  void GetXlaBatchSize();
  void* GetAddrOfTensor(const Tensor* ptr);
  size_t GetSizeOfTensor(const Tensor* ptr);
  int GetBatchsizeToPad(const int origin_batchsize);
  void GetPaddedInputs(const std::vector<std::pair<std::string, Tensor>>& origin_input, 
    std::vector<std::pair<std::string, Tensor>>& padded_input);
  void GetRealOutputs(const std::vector<Tensor>& padded_output, 
    std::vector<Tensor>& real_output, int real_batchsize);
  void WriteOutputsToFile(FILE* fp, const std::vector<Tensor>& outputs);

  struct PredictContext {
    Session* session;
    bool borrowed;
    Model* parent;
    int ctxid;
  };
  PredictContext* Borrow();
  void Return(PredictContext* context);

  const std::string& name() const { return name_; }
  const SessionOptions& sess_options() const {
    return sess_options_;
  }
  const RunOptions& run_options() const {
    return run_options_;
  }
  const std::vector<std::pair<std::string, Tensor>>& GetOneInput(int* batchsize) const {
    static int idx = 0;
    int temp = idx++ % inputs_.size();
    *batchsize = inferred_batchsizes_[temp];
    return inputs_[temp];
  }
  const std::vector<std::string>& output_names() const {
    return output_names_;
  }
  const std::vector<PredictContext>& predict_contexts() const {
    return predict_contexts_;
  }

 protected:
  std::string name_;
  int predictor_num_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::vector<PredictContext> predict_contexts_;
  GraphDef gdef_;
  SessionOptions sess_options_;
  RunOptions run_options_;

  std::vector<std::vector<std::pair<std::string, Tensor>>> inputs_;
  std::vector<int> inferred_batchsizes_;

  std::vector<std::string> output_names_;
  std::unordered_map<std::string, Session*> sessions_;
  std::string graph_path_;
  int batchsize_internal_;
  std::vector<int> xla_batchsize_;
  std::mutex batchsize_mtx_;
};

class ModelReloader : public DoubleBufferReloader<Model> {
 public:
  ModelReloader(const benchmark::BenchModelConfig& bench_model_config)
      : bench_model_config_(bench_model_config) {}

  virtual Model* CreateObject() override;

 private:
  benchmark::BenchModelConfig bench_model_config_;
};

class ModelSelector {
 public:
  ModelSelector() {}
  virtual ~ModelSelector() = default;

  void Start();
  void Stop();

  bool InitModel(const benchmark::BenchModelConfig& bench_model_config);
  std::shared_ptr<Model> GetModel(int idx) const;

 private:
  std::vector<std::shared_ptr<ModelReloader>> model_reloaders_;
  std::vector<int> switch_interval_;
  std::atomic_bool running_;
};

}  // namespace benchmark
