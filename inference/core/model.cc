#include "benchmark/core/model.h"
#include "benchmark/proto/sample.pb.h"
#include "benchmark/core/node_util.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

using namespace tensorflow;
namespace benchmark {

void Model::AddDeepCopyNode() {
  std::set<std::string> inode_name_set;
  for (auto& kv : inputs_[0]) {
    inode_name_set.emplace(kv.first);
  }
  GraphDef origin_gdef;
  for (auto& node : gdef_.node()) {
    *(origin_gdef.add_node()) = node;
  }
  gdef_.Clear();
  for (auto& node : origin_gdef.node()) {
    *(gdef_.add_node()) = node;
    // If the node is input node, then add a DeepCopy node.
    if (inode_name_set.count(node.name())) {
      NodeDef dp_node;
      AttrValue dtype_attr;
      SetAttrValue(node.attr().at("dtype"), &dtype_attr);
      dp_node.mutable_attr()->insert({"T", dtype_attr});
      dp_node.set_device("/device:CPU:0");
      dp_node.set_op("DeepCopy");
      dp_node.set_name(node.name() + "/DeepCopy");
      // establish connection
      dp_node.add_input(node.name());
      *(gdef_.add_node()) = dp_node;
    }
  }
  // Placeholder -> other, Placeholder -> DeepCopy -> other.
  for (auto& node : *(gdef_.mutable_node())) {
    for (auto& input_tensor_name : *(node.mutable_input())) {
      // If tensor name is `node:1`, then node name is `node`.
      auto input_node_name = NodeNameFromInput(input_tensor_name);
      if (inode_name_set.count(input_node_name)) {
        auto dp_node_name = input_node_name + "/DeepCopy";
        // The node cannot be a DeepCopy node.
        if (node.name() != dp_node_name) input_tensor_name = dp_node_name;
      }
    }
  }
}

int Model::GetNumOfVirtualGPU(ConfigProto* config) {
  auto iter = config->device_count().find("GPU");
  CHECK(iter != config->device_count().end() && iter->second == 1);
  auto gpu_options = config->gpu_options();
  CHECK(gpu_options.visible_device_list() == "0");
  auto virtual_devs = gpu_options.experimental().virtual_devices();
  CHECK(virtual_devs.size() == 1);
  return virtual_devs.at(0).memory_limit_mb_size();
}

void Model::PlaceGraphToDevice(int gpu_idx) {
  auto gpu_name = "/device:GPU:" + std::to_string(gpu_idx);
  for (auto& node : *(gdef_.mutable_node())) {
    if (node.op() != "DeepCopy") node.set_device(gpu_name);
  }
}

void Model::GetXlaBatchSize() {
  // Calculate the range of xla batchsize.
  int min_bs = inferred_batchsizes_.at(0), max_bs = inferred_batchsizes_.at(0);
  for (auto& val : inferred_batchsizes_) {
    if (val < min_bs) min_bs = val;
    if (val > max_bs) max_bs = val;
  }
  auto min_bs_str = std::to_string(min_bs), max_bs_str = std::to_string(max_bs);
  auto xla_min_bs = min_bs_str[0] - '0', xla_max_bs = max_bs_str[0] - '0' + 1;
  for (int i = 0; i < min_bs_str.size() - 1; i++) {
    xla_min_bs *= 10;
  }
  for (int i = 0; i < max_bs_str.size() - 1; i++) {
    xla_max_bs *= 10;
  }
  for (int bs = xla_min_bs; bs <= xla_max_bs; bs += batchsize_internal_) {
    if (bs != xla_min_bs) xla_batchsize_.push_back(bs);
  }
  LOG(INFO) << "The number of xla batchsize is " << xla_batchsize_.size();
}

void* Model::GetAddrOfTensor(const Tensor* ptr) {
  switch (ptr->dtype()) {
    case DT_HALF:
      return (void*)ptr->flat<EnumToDataType<DT_HALF>::Type>().data();
    case DT_FLOAT:
      return (void*)ptr->flat<EnumToDataType<DT_FLOAT>::Type>().data();
    case DT_INT8:
      return (void*)ptr->flat<EnumToDataType<DT_INT8>::Type>().data();
    case DT_INT32:
      return (void*)ptr->flat<EnumToDataType<DT_INT32>::Type>().data();
    default:
      LOG(ERROR) << DataTypeString(ptr->dtype()) << "is not supported.";
      return nullptr;
  }
}

size_t Model::GetSizeOfTensor(const Tensor* ptr) {
  switch (ptr->dtype()) {
    case DT_HALF:
      return ptr->flat<EnumToDataType<DT_HALF>::Type>().size() * sizeof(EnumToDataType<DT_HALF>::Type);
    case DT_FLOAT:
      return ptr->flat<EnumToDataType<DT_FLOAT>::Type>().size() * sizeof(EnumToDataType<DT_FLOAT>::Type);
    case DT_INT8:
      return ptr->flat<EnumToDataType<DT_INT8>::Type>().size() * sizeof(EnumToDataType<DT_INT8>::Type);
    case DT_INT32:
      return ptr->flat<EnumToDataType<DT_INT32>::Type>().size() * sizeof(EnumToDataType<DT_INT32>::Type);
    default:
      LOG(ERROR) << DataTypeString(ptr->dtype()) << "is not supported.";
      return 0;
  }
}

int Model::GetBatchsizeToPad(const int origin_batchsize) {
  for (auto& val : xla_batchsize_) {
    if (val >= origin_batchsize) return val;
  }
  auto bs_str = std::to_string(origin_batchsize);
  int pad_batchsize = bs_str[0] - '0' + 1;
  for (int i = 0; i < bs_str.size() - 1; i++) {
    pad_batchsize *= 10;
  }
  std::unique_lock<std::mutex> locker(batchsize_mtx_);
  xla_batchsize_.emplace_back(pad_batchsize);
  locker.unlock();
  return pad_batchsize;
}

void Model::GetPaddedInputs(const std::vector<std::pair<std::string, Tensor>>& origin_input, 
    std::vector<std::pair<std::string, Tensor>>& padded_input) {
  padded_input.reserve(origin_input.size());
  for (int i = 0; i < origin_input.size(); i++) {
    auto& src_tensor = origin_input[i].second;
    auto& src_shape = src_tensor.shape();
    int origin_batchsize = src_shape.dim_size(0);
    int pad_batchsize = GetBatchsizeToPad(origin_batchsize);
    if (origin_batchsize == 1 || pad_batchsize == origin_batchsize) {
      padded_input.emplace_back(std::make_pair(origin_input[i].first, src_tensor));
    } else {
      TensorShape dst_shape = src_shape;
      dst_shape.set_dim(0, pad_batchsize);
      // use default_cpu_allocator
      Tensor dst_tensor(src_tensor.dtype(), dst_shape);
      void* src_addr = GetAddrOfTensor(&src_tensor);
      void* dst_addr = GetAddrOfTensor(&dst_tensor);
      size_t src_size = GetSizeOfTensor(&src_tensor);
      size_t dst_size = GetSizeOfTensor(&dst_tensor);
      memset(dst_addr, 0, dst_size);
      memcpy(dst_addr, src_addr, src_size);
      padded_input.emplace_back(std::make_pair(origin_input[i].first, std::move(dst_tensor)));
    }
  }
}

void Model::GetRealOutputs(const std::vector<Tensor>& padded_output, 
    std::vector<Tensor>& real_output, int real_batchsize) {
  real_output.reserve(padded_output.size());
  for (int i = 0; i < padded_output.size(); i++) {
    auto& padded_shape = padded_output[i].shape();
    if (padded_shape.dim_size(0) == real_batchsize) {
      real_output.emplace_back(std::move(padded_output[i]));
    } else {
      real_output.emplace_back(padded_output[i].Slice(0, real_batchsize));
    }
  }
}

void Model::WriteOutputsToFile(FILE* fp, const std::vector<Tensor>& outputs) {
  float* our_addr = (float*)GetAddrOfTensor(&outputs[0]);
  int dim0size = outputs[0].dim_size(0), dim1size = outputs[0].dim_size(1);
  for (int idx = 1; idx < dim0size * dim1size; idx+=2) {
    fprintf(fp, "%e ", our_addr[idx]);
  }
  fprintf(fp, "\n");
}

Model::PredictContext *Model::Borrow() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& context : predict_contexts_) {
    if (!context.borrowed) {
      context.borrowed = true;
      return &context;
    }
  }
  auto timeout = std::chrono::microseconds(500);
  PredictContext *res = nullptr;
  auto pred = [&]() {
    for (auto& context : predict_contexts_) {
      if (!context.borrowed) {
        context.borrowed = true;
        res = &context;
        return true;
      }
    }
    return false;
  };
  if (cond_.wait_for(lock, timeout, pred)) {
    return res;
  }
  return nullptr;
}

void Model::Return(PredictContext *predict_context) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto& context : predict_contexts_) {
    if (context.session == predict_context->session) {
      context.borrowed = false;
    }
  }
  cond_.notify_one();
}

bool Model::ParseRunOptions(const std::string& run_options) {
  if (run_options.empty()) {
    LOG(WARNING) << "No run_options path configured: " << name()
                 << ", use default RunOptions.";
    return false;
  }
  Status s = ReadTextProto(Env::Default(), run_options.c_str(), &run_options_);
  if (!s.ok()) {
    s = ReadBinaryProto(Env::Default(), run_options.c_str(), &run_options_);
    if (!s.ok()) {
      LOG(ERROR) << "Read run options failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  VLOG(1) << "Read run options, " << run_options << ": " << run_options_.DebugString();
  return true;
}

bool Model::InitSession(const std::string& config_proto) {

  if (inputs_.empty()) {
    LOG(ERROR) << "No vaild inputs: " << name();
    return false;
  }

  // Prepare Session ConfigProto
  Status s;
  SessionOptions session_options = SessionOptions();
  ConfigProto* config = &session_options.config;
  if (!config_proto.empty()) {
    s = ReadTextProto(Env::Default(), config_proto.c_str(), config);
    if (!s.ok()) {
      s = ReadBinaryProto(Env::Default(), config_proto.c_str(), config);
      if (!s.ok()) {
        LOG(ERROR) << "Read config proto failed: " << name() << ", " << s.ToString()
                   << ". Use default ConfigProto.";
      }
    }
    VLOG(1) << "Read config proto: " << name() << ", " << config->DebugString();
  }
  config->set_allow_soft_placement(true);

  int num_gpus = GetNumOfVirtualGPU(config);
  LOG(INFO) << "The number of virtual GPUs is " << num_gpus;
  // Allocate pinned memory for input nodes.
  AddDeepCopyNode();
  for (int i = 0; i < this->predictor_num_; ++i) {
    int gpu_idx = i % num_gpus;
    Session* session = nullptr;
    std::string session_key = name() + "/GPU:" + std::to_string(gpu_idx);
    auto iter = sessions_.find(session_key);
    if (iter == sessions_.end()) {
      PlaceGraphToDevice(gpu_idx);
      s = NewSession(session_options, &session);
      if (!s.ok()) {
        LOG(ERROR) << "New session failed: " << name() << ", " << s.ToString();
        return false;
      }
      s = session->Create(gdef_);
      if (!s.ok()) {
        LOG(ERROR) << "Create session failed: " << name() << ", " << s.ToString();
        return false;
      }
      sessions_[session_key] = session;
    } else {
      session = iter->second;
    }
    PredictContext context{session, false, this, i};
    predict_contexts_.push_back(context);
    LOG(INFO) << "Predictor " << i << " uses session " << session_key;
  }
  return true;
}

bool Model::ParseSamples(const std::string& sample_file) {
  if (sample_file.empty()) {
    LOG(ERROR) << "Samplefile path must not be empty: " << name();
    return false;
  }
  SamplesProto samples_proto;
  Status s = ReadBinaryProto(Env::Default(), sample_file.c_str(), &samples_proto);
  if (!s.ok()) {
    s = ReadTextProto(Env::Default(), sample_file.c_str(), &samples_proto);
    if (!s.ok()) {
      LOG(ERROR) << "Read sample_file failed: " << name() << ", " << s.ToString();
      return false;
    }
  }
  VLOG(2) << "Samples_proto: " << samples_proto.DebugString();

  for (int i = 0; i < samples_proto.output_names_size(); ++i) {
    output_names_.push_back(samples_proto.output_names(i));
  }
  
  inputs_.clear();
  inputs_.reserve(samples_proto.sample_size());

  for (int i = 0; i < samples_proto.sample_size(); ++i) {
    const InputsProto& inputs_proto = samples_proto.sample(i);
    std::vector<std::pair<std::string, Tensor>> sample_vec;
    int64 batchsize = 1;
    for (int j = 0; j < inputs_proto.input_size(); ++j) {
      const NamedTensorProto& input = inputs_proto.input(j);
      Tensor tensor;
      if (!tensor.FromProto(input.tensor())) { 
        LOG(ERROR) << "Init tensor from proto failed.";
        return false;
      }
      sample_vec.emplace_back(input.name(), tensor);
      if (tensor.dims() >=1 && tensor.dim_size(0) > batchsize)
        batchsize = tensor.dim_size(0);
    }
    inputs_.emplace_back(std::move(sample_vec));
    inferred_batchsizes_.emplace_back(batchsize);
    LOG(INFO) << "Parsed input, inferred batchsize = " << batchsize;
  }

  LOG(INFO) << "Parse input_samples success, total "<<inputs_.size()<<"samples";

  return true;
}

bool Model::LoadGraph(const std::string& frozen_graph) {
  if (frozen_graph.empty()) {
    LOG(ERROR) << "No graph path configured: " << name();
    return false;
  }
  Status s = ReadTextProto(Env::Default(), frozen_graph.c_str(), &gdef_);
  if (!s.ok()) {
    s = ReadBinaryProto(Env::Default(), frozen_graph.c_str(), &gdef_);
    if (!s.ok()) {
      LOG(ERROR) << "Read graph failed: " << name() << ", " << s.ToString();
      return false;
    }
  }

  graph_path_ = frozen_graph;
  size_t pos = graph_path_.find_last_of("/");
  if (pos != std::string::npos) {
    graph_path_ = graph_path_.substr(0, pos + 1);
  } else {
    graph_path_ = "./";
  }
  return true;
}

bool Model::Warmup() {
  for (auto context : predict_contexts_) {
    Session* session = context.session;
    FILE* fp = nullptr;
    if (context.ctxid == 0) fp = fopen("score.txt", "w");
    for (int i = 0; i < inputs_.size(); i++) {
      // Get padded inputs from real inputs.
      std::vector<std::pair<std::string, Tensor>> padded_inputs;
      GetPaddedInputs(inputs_[i], padded_inputs);
      std::vector<Tensor> padded_outputs;
      RunMetadata meta;
      Status our_s = session->Run(run_options_, padded_inputs, output_names_, {}, &padded_outputs, &meta);
      if (!our_s.ok()) {
        LOG(ERROR) << "Warmup: " << name() << ", Session::Run failed: " << i
                   << ", inferred_batchsize = " << inferred_batchsizes_[i] << ", " << our_s.ToString();
        return false;
      }
      // Get real outputs from padded outputs.
      std::vector<Tensor> real_outputs;
      GetRealOutputs(padded_outputs, real_outputs, inferred_batchsizes_[i]);
      // Write real outputs to "score.txt".
      if (context.ctxid == 0) WriteOutputsToFile(fp, real_outputs);
    }
    if (context.ctxid == 0) fclose(fp);
  }
  return true;
}

Model* ModelReloader::CreateObject() {
  Model *model= new Model(bench_model_config_.name(), bench_model_config_.predictor_num());
  // Load graph
  if (!model->LoadGraph(bench_model_config_.frozen_graph())) {
    LOG(ERROR) << "Load graph failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  if (!model->ParseSamples(bench_model_config_.sample_file())) {
    LOG(ERROR) << "Read sample_file failed: " << bench_model_config_.sample_file() << "," 
               << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Init TensorFlow Session
  if (!model->InitSession(bench_model_config_.config_proto())) {
    LOG(ERROR) << "Init tensorflow session failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  // Prepare Session RunOptions
  if (!model->ParseRunOptions(bench_model_config_.run_options())) {
    LOG(ERROR) << "Parse run options failed: " << bench_model_config_.name()
               << ", use default RunOptions.";
  }

  // Warmup
  if (!model->Warmup()) {
    LOG(ERROR) << "Warmup failed: " << bench_model_config_.name();
    delete model;
    return nullptr;
  }

  LOG(INFO) << "Init and warmup model complete: " << bench_model_config_.name();
  return model;
}

bool ModelSelector::InitModel(
    const benchmark::BenchModelConfig& bench_model_config) {
  std::shared_ptr<ModelReloader> model_reloader =
      std::make_shared<ModelReloader>(bench_model_config);
  bool success = model_reloader->Switch();
  if (!success) {
    return false;
  }
  model_reloaders_.emplace_back(model_reloader);
  switch_interval_.emplace_back(bench_model_config.switch_interval());
  return true;
}

std::shared_ptr<Model> ModelSelector::GetModel(int idx) const {
  auto model_reloader = model_reloaders_[idx];
  return model_reloader->Instance();
}

void ModelSelector::Start() {
  running_ = true;
  std::vector<int> left_time_to_switch(switch_interval_);
  while (running_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    for (int i = 0; i < left_time_to_switch.size(); ++i) {
      left_time_to_switch[i]--;
      if (left_time_to_switch[i] <= 0) {
        LOG(INFO) << "Begin switch model.";
        bool success = model_reloaders_[i]->Switch();
        if (!success) {
          LOG(ERROR) << "Switch model failed.";
          continue;
        }
        LOG(INFO) << "Switch model successfully.";
        left_time_to_switch[i] = switch_interval_[i];
      }
    }
  }
}

void ModelSelector::Stop() { running_ = false; }

}  // namespace benchmark
