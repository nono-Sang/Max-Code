#include <fstream>
#include "benchmark/core/predict_request_consumer.h"

using namespace tensorflow;
namespace benchmark {

PredictRequestConsumer::PredictRequestConsumer(
    benchmark::ModelSelector *model_selector,
    benchmark::PredictRequestQueue *predict_queue,
    benchmark::Metrics *metrics, int max_queue_size) {
  model_selector_ = model_selector;
  predict_queue_ = predict_queue;
  metrics_ = metrics;
  max_queue_size_ = max_queue_size;
}

void PredictRequestConsumer::Start() {
  while (!metrics_->IsStopped()) {
    PredictRequest *predict_request = predict_queue_->Dequeue();
    if (!predict_request) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }

    int model_idx = predict_request->model_idx;
    std::shared_ptr<Model> model = model_selector_->GetModel(model_idx);
    if (!model) {
      LOG(ERROR) << "model_idx: " << model_idx << " out of range.";
      return;
    }
    if (max_queue_size_ > 0 && predict_queue_->size() > max_queue_size_) {
      metrics_->UpdateGetPredictorFailures(model->name());
      VLOG(2) << "Drop request: number of outstanding requests exceeds max_queue_size.";
      continue;
    }
    Model::PredictContext* predict_context = model->Borrow();
    if (!predict_context) {
      predict_queue_->Enqueue(predict_request);
      continue;
    }
    auto bef = std::chrono::high_resolution_clock::now();
    int batchsize = 0;
    if (this->PredictImpl(predict_context, &batchsize)) {
      metrics_->UpdateThroughput(model->name());
    } else {
      metrics_->UpdateFailures(model->name());
    }
    auto aft = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(aft - bef).count();
    metrics_->UpdateLatency(model->name(), dur);
    metrics_->UpdateBatchsize(model->name(), batchsize);
    model->Return(predict_context);
  }
}

bool PredictRequestConsumer::PredictImpl(
    benchmark::Model::PredictContext* predict_context, int* batchsize) {
  auto model = predict_context->parent;
  auto model_name = model->name();
  auto session = predict_context->session;
  // Get origin_inputs and padded_inputs.
  auto origin_inputs = model->GetOneInput(batchsize);
  std::vector<std::pair<std::string, Tensor>> padded_inputs;
  model->GetPaddedInputs(origin_inputs, padded_inputs);
  std::vector<Tensor> padded_outputs;
  RunMetadata meta;
  auto s = session->Run(model->run_options(), padded_inputs,
                        model->output_names(), {}, &padded_outputs, &meta);
  if (!s.ok()) {
    LOG(ERROR) << model_name << ", Session::Run failed: " << s.ToString();
    return false;
  }
  if (model->run_options().trace_level() > RunOptions::NO_TRACE) {
    static std::mutex mu;
    std::lock_guard<std::mutex> guard(mu);
    auto path = model_name + ".runmeta";
    std::ofstream dump;
    dump.open(path, std::ofstream::out | std::ofstream::trunc);
    if (!meta.SerializeToOstream(&dump)) {
      LOG(ERROR) << model_name << ", dump trace file failed.";
      return false;
    }
    dump.close();
  }

  // Get real_outputs from padded_outputs.
  std::vector<Tensor> real_outputs;
  model->GetRealOutputs(padded_outputs, real_outputs, *batchsize);
  std::vector<std::string> output_names = model->output_names();
  if (real_outputs.size() != output_names.size()) {
    LOG(ERROR) << model_name << ", output numbers mismatch.";
    return false;
  }
  for (int i = 0; i < real_outputs.size(); i++) {
    TensorProto proto;
    real_outputs[i].AsProtoField(&proto);
    VLOG(1) << model_name << ", output " << output_names[i] << " (output of session::run): "<< proto.DebugString();
  }
  return true;
}

}  // namespace benchmark
