#include <bits/stdc++.h>
#include <grpcpp/grpcpp.h>
#include <hdfs.h>

#include <boost/asio.hpp>
#include <etcd/Client.hpp>
#include <etcd/Response.hpp>

#include "alimama.grpc.pb.h"
#include "utils.h"
#include "ModelSliceReader.h"

using alimama::proto::InterNodeService;
using alimama::proto::ModelService;
using alimama::proto::Request;
using alimama::proto::Response;
using grpc::Channel;
using grpc::ChannelArguments;
using grpc::ClientContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ResourceQuota;

const std::string hdfs_url = "hdfs://namenode:9000";
const std::string etcd_url = "http://etcd:2379";
const std::string local_file_path = "/modeldata/";
const std::string inter_node_path = "/InterNode/";
const std::string register_path = "/services/modelservice/";
const int model_name_len = 25;
const int total_node_num = 6;
const int max_model_count = 2;
const int check_model_internal = 10;  // seconds (magic!)
const int check_done_internal = 1;  // seconds


struct FixStoreInfo {
  char* local_buf_;
  std::unordered_map<int, char*> part_buf_map_;
};

struct ModelSliceInfo {
  std::string model_name_;
  char* local_buf_;
  std::unordered_map<int, char*>& part_buf_map_;
  explicit ModelSliceInfo(const std::string& name, 
    char* buf, std::unordered_map<int, char*>& map)
  : model_name_(name), local_buf_(buf), part_buf_map_(map) {}
};

class FetchClient {
 public:
  explicit FetchClient(std::shared_ptr<Channel> channel)
  : stub_(InterNodeService::NewStub(channel)) {}

  Response* FetchRemoteSlice(Request& node_req) {
    Response* res_ptr = new Response;
    ClientContext context;
    Status status = stub_->Fetch(&context, node_req, res_ptr);
    if(status.ok() == false) {
      LOG_INFO("error code is %s.", status.error_message().c_str()); exit(-1);
    }
    return res_ptr;
  }

 private:
  std::unique_ptr<InterNodeService::Stub> stub_;
};

class GlobalServer final {
  friend class ModelServiceImpl;
  friend class InterNodeServiceImpl;
 public:
  explicit GlobalServer(int node_id);
  ~GlobalServer();
  void RunGlobalServer();
  std::string GetLatestModel();
  void WaitModelDone(std::string model_name);
  void ModelAnalysis(std::string model_name);
  void LoadPartModel(std::string model_name, bool init_model);
  void BuildFetchServer();
  void BuildGetServer();
  void InitFetchClient();

 private:
  int node_id_;
  int version_cnt_;
  std::string local_ip_;
  hdfsFS fs_;
  etcd::Client etcd_client_;
  ModelSliceReader slice_reader_;
  std::thread fetch_th_;
  std::thread get_th_;
  std::unordered_map<int, FetchClient*> node_client_map_;
  std::vector<std::thread> th_pool_;
  boost::asio::io_context pool_ctx_;
  boost::asio::io_context::work pool_work_;

  uint64_t each_slice_size_;
  std::map<int, int> part_node_map_;
  std::map<int, std::string> part_name_map_;
  std::vector<int> local_part_vec_;
  FixStoreInfo store_info_arr_[max_model_count];
  std::shared_mutex model_info_mtx_;
  std::vector<ModelSliceInfo*> model_info_vec_;
};

class ModelServiceImpl final : public ModelService::Service {
 public:
  explicit ModelServiceImpl(GlobalServer* gs_ptr): gs_ptr_(gs_ptr) {}
  Status Get(ServerContext* ctx, const Request* request, Response* response) override;
 private:
  GlobalServer* gs_ptr_;
};

class InterNodeServiceImpl final : public InterNodeService::Service {
 public:
  explicit InterNodeServiceImpl(GlobalServer* gs_ptr): gs_ptr_(gs_ptr) {}
  Status Fetch(ServerContext* ctx, const Request* request, Response* response) override;
 private:
  GlobalServer* gs_ptr_;
};


GlobalServer::GlobalServer(int node_id)
 : node_id_(node_id), version_cnt_(0), etcd_client_(etcd_url), pool_work_(pool_ctx_) {
  GetLocalIp(local_ip_);
  hdfsBuilder *builder = hdfsNewBuilder();
  hdfsBuilderSetNameNode(builder, hdfs_url.c_str());
  fs_ = hdfsBuilderConnect(builder);
  CHECK(fs_ != nullptr);

  fetch_th_ = std::thread([this] { BuildFetchServer(); });
  get_th_ = std::thread([this] { BuildGetServer(); });
  th_pool_.resize(total_node_num - 1);
  for (int i = 0; i < total_node_num - 1; i++) {
    th_pool_[i] = std::thread([this] { pool_ctx_.run(); });
  }
}

GlobalServer::~GlobalServer() {
  hdfsDisconnect(fs_);
  get_th_.join();
  fetch_th_.join();
  pool_ctx_.stop();
  for (auto& th : th_pool_) th.join();
  for (auto& kv : node_client_map_) delete kv.second;
  for (auto& val : model_info_vec_) delete val;
  free(store_info_arr_[0].local_buf_);
  free(store_info_arr_[1].local_buf_);
}

void GlobalServer::RunGlobalServer() {
  std::string init_model = GetLatestModel();
  WaitModelDone(init_model);
  ModelAnalysis(init_model);
  LoadPartModel(init_model, true);
  InitFetchClient();
  std::string key = register_path + local_ip_ + ":1024";
  auto rep = etcd_client_.set(key, std::to_string(node_id_)).get();
  CHECK(rep.is_ok());

  while (true) {
    sleep(check_model_internal);
    std::string latest_model = GetLatestModel();
    bool exist = false;
    std::unique_lock<std::shared_mutex> locker(model_info_mtx_);
    for (auto& val : model_info_vec_) {
      if (val->model_name_ == latest_model) exist = true;
    }
    locker.unlock();
    if (!exist) {
      WaitModelDone(latest_model);
      LoadPartModel(latest_model, false);
    }
  }
}

std::string GlobalServer::GetLatestModel() {
  int num_all = 0;
  hdfsFileInfo* all_info = nullptr;
  std::vector<std::string> model_vec;
  while (true) {
    all_info = hdfsListDirectory(fs_, "/", &num_all);
    for (int i = 0; i < num_all; i++) {
      // If there is a rollback, return directly.
      int rb_pos = std::string(all_info[i].mName).find("rollback");
      if (rb_pos != -1) {
        hdfsFile rb = hdfsOpenFile(fs_, "/rollback.version", O_RDONLY, 0, 0, 0);
        CHECK(rb != nullptr);
        char name_arr[model_name_len];
        hdfsRead(fs_, rb, name_arr, model_name_len);
        CHECK(hdfsCloseFile(fs_, rb) == 0);
        hdfsFreeFileInfo(all_info, num_all);
        return std::string(name_arr);
      }
      int pos = std::string(all_info[i].mName).find("model_");
      if (pos != -1) {
        std::string model_name = std::string(all_info[i].mName).substr(pos);
        model_vec.emplace_back(model_name);
      }
    }
    hdfsFreeFileInfo(all_info, num_all);
    if (model_vec.size()) break;  // always occur except init
  }
  std::sort(model_vec.begin(), model_vec.end());
  return model_vec[model_vec.size() - 1];
}

void GlobalServer::WaitModelDone(std::string model_name) {
  std::string done_path = "/" + model_name + "/model.done";
  hdfsFileInfo *done_info = nullptr;
  while (true) {
    done_info = hdfsGetPathInfo(fs_, done_path.c_str());
    if (done_info != nullptr) break;
    else sleep(check_done_internal);
  }
  hdfsFreeFileInfo(done_info, 1);
  LOG_INFO("%s is ready.", model_name.c_str());
}

void GlobalServer::ModelAnalysis(std::string model_name) {
  auto local_model_path = local_file_path + model_name + "/";
  auto remote_meta_path = "/" + model_name + "/model.meta";
  Command("mkdir -p " + local_model_path);
  Command("hdfs dfs -fs hdfs://namenode:9000/  -get " 
          + remote_meta_path + " " + local_model_path);
  
  auto local_meta_path = local_model_path + "model.meta";
  std::ifstream file_fd(local_meta_path);
  int total_slice_num = 0;
  std::string line;
  std::vector<int> all_part_vec;
  getline(file_fd, line);
  while (getline(file_fd, line)) {
    if (!line.size()) continue;
    int pos1 = line.find("model_slice.");
    int pos2 = line.find(",size:");
    CHECK(pos1 != -1 && pos2 != -1);
    int part = atoi(line.substr(pos1 + 12, pos2 - pos1 - 12).c_str());
    if (total_slice_num == 0) {
      each_slice_size_ = atoi(line.substr(pos2 + 6).c_str());
    }
    all_part_vec.emplace_back(part);
    part_name_map_.insert({part, line.substr(pos1, pos2 - pos1)});
    total_slice_num += 1;
  }
  file_fd.close();

  int num_slice_each_node = total_slice_num / total_node_num;
  for (int i = 0; i < total_slice_num; i++) {
    if (i < total_node_num * num_slice_each_node) {
      part_node_map_.insert({
        all_part_vec.at(i), i / num_slice_each_node + 1});
    } else {
      part_node_map_.insert({
        all_part_vec.at(i), (i + 1) % total_node_num});
    }
  }
  for (auto& kv : part_node_map_) {
    if (kv.second == node_id_) local_part_vec_.emplace_back(kv.first);
  }

  // Allocate memory in advance.
  int local_slice_num = local_part_vec_.size();
  uint64_t total_buf_size = each_slice_size_ * local_part_vec_.size();
  char* addr = reinterpret_cast<char*>(malloc(total_buf_size * max_model_count));
  store_info_arr_[0].local_buf_ = addr;
  store_info_arr_[1].local_buf_ = addr + total_buf_size;
  char* oft_0 = store_info_arr_[0].local_buf_;
  char* oft_1 = store_info_arr_[1].local_buf_;
  for (int i = 0; i < local_slice_num; i++) {
    int part = local_part_vec_.at(i);
    store_info_arr_[0].part_buf_map_.insert({part, oft_0});
    store_info_arr_[1].part_buf_map_.insert({part, oft_1});
    oft_0 += each_slice_size_;
    oft_1 += each_slice_size_;
  }
}

void GlobalServer::LoadPartModel(std::string model_name, bool init_model) {
  auto local_model_path = local_file_path + model_name + "/";
  if (init_model == false) {
    Command("mkdir -p " + local_model_path);
  }

  int local_slice_num = local_part_vec_.size();
  std::vector<std::promise<bool>> prom_vec(local_slice_num);
  std::vector<std::future<bool>> futu_vec(local_slice_num);
  for (int i = 0; i < local_slice_num; i++) {
    futu_vec[i] = prom_vec[i].get_future();
  }

  pool_ctx_.post([this, &prom_vec, &model_name, &local_model_path] {
    int local_slice_num = local_part_vec_.size();
    for (int i = 0; i < local_slice_num; i++) {
      auto part_name = part_name_map_.at(local_part_vec_[i]);
      auto hdfs_part_path = "/" + model_name + "/" + part_name;
      auto local_part_path = local_model_path + part_name;
      Command("hdfs dfs -fs hdfs://namenode:9000/  -get "
              + hdfs_part_path + " " + local_part_path);
      prom_vec[i].set_value(true);
    }
  });

  version_cnt_ += 1;
  ModelSliceInfo* new_model = nullptr;
  if (version_cnt_ == 1 || version_cnt_ == 2) {
    int idx = version_cnt_ - 1;
    new_model = new ModelSliceInfo(model_name, 
      store_info_arr_[idx].local_buf_, 
      store_info_arr_[idx].part_buf_map_);
  } else {
    std::lock_guard<std::shared_mutex> locker(model_info_mtx_);
    CHECK(model_info_vec_.size() == max_model_count);
    auto info = model_info_vec_.at(0);
    new_model = new ModelSliceInfo(model_name, 
      info->local_buf_, info->part_buf_map_);
    delete model_info_vec_.at(0);
    model_info_vec_.erase(model_info_vec_.begin());
  }

  for (int i = 0; i < local_slice_num; i++) {
    int part = local_part_vec_[i];
    std::string part_name = part_name_map_.at(part);
    std::string local_part_path = local_model_path + part_name;
    char* offset = new_model->part_buf_map_.at(part);
    CHECK(futu_vec[i].get());
    slice_reader_.Load(local_part_path);
    slice_reader_.Read(0, each_slice_size_, offset);
    slice_reader_.Unload();
  }
  LOG_INFO("Finished loading %s.", model_name.c_str());
  Command("rm -r -f " + local_model_path);

  // Synchronize all nodes and update model vector.
  std::string prefix = "/" + model_name + std::to_string(version_cnt_) + "/";
  auto response = etcd_client_.set(prefix + std::to_string(node_id_), "").get();
  CHECK(response.is_ok());
  while (true) {
    auto res = etcd_client_.ls(prefix).get();
    CHECK(res.is_ok());
    if (res.keys().size() == total_node_num) break;
  }

  {
    std::lock_guard<std::shared_mutex> locker(model_info_mtx_);
    model_info_vec_.emplace_back(new_model);
  }
  LOG_INFO("All nodes have synchronized.");
}

void GlobalServer::BuildFetchServer() {
  std::string server_addr = local_ip_ + ":4096";
  InterNodeServiceImpl service(this);
  ServerBuilder builder;
  builder.SetSyncServerOption(ServerBuilder::NUM_CQS, total_node_num - 1);
  builder.SetSyncServerOption(ServerBuilder::MAX_POLLERS, total_node_num - 1);
  builder.AddListeningPort(server_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 3 * 60 * 1000);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 3 * 60 * 1000);
  builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG_INFO("Fetch server listening on %s", server_addr.c_str());
  server->Wait();
}

void GlobalServer::BuildGetServer() {
  std::string server_addr = local_ip_ + ":1024";
  ModelServiceImpl service(this);
  ServerBuilder builder;
  builder.AddListeningPort(server_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  builder.SetSyncServerOption(ServerBuilder::NUM_CQS, total_node_num - 1);
  builder.SetSyncServerOption(ServerBuilder::MAX_POLLERS, total_node_num - 1);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG_INFO("Get server listening on %s", server_addr.c_str());
  server->Wait();
}

void GlobalServer::InitFetchClient() {
  std::string server_addr = local_ip_ + ":4096";
  std::string key = inter_node_path + std::to_string(node_id_);
  auto rep = etcd_client_.set(key, server_addr).get();
  CHECK(rep.is_ok());
  while (true) {
    auto lsrep = etcd_client_.ls(inter_node_path).get();
    int lsnum = lsrep.keys().size();
    if (lsnum == total_node_num) {
      for (int nid = 1; nid <= total_node_num; nid++) {
        if (nid == node_id_) continue;
        auto rep = etcd_client_.get(inter_node_path + std::to_string(nid)).get();
        CHECK(rep.is_ok());
        auto addr = rep.value().as_string();
        LOG_INFO("node %d fetch server is %s.", nid, addr.c_str());
        ChannelArguments args;
        args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 3* 60 * 1000);
        args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 3 * 60 * 1000);
        args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
        auto client_ptr = new FetchClient(grpc::CreateCustomChannel(
          addr, grpc::InsecureChannelCredentials(), args));
        node_client_map_.insert({nid, client_ptr});
      }
      break;
    } else sleep(1);
  }
  LOG_INFO("Connections have been established between nodes.");
}

Status ModelServiceImpl::Get(ServerContext* ctx, const Request* request, Response* response) {
  ModelSliceInfo* info_ptr = nullptr;
  std::shared_lock<std::shared_mutex> locker(gs_ptr_->model_info_mtx_);
  int len = gs_ptr_->model_info_vec_.size();
  CHECK(len);
  info_ptr = gs_ptr_->model_info_vec_[len - 1];
  locker.unlock();

  int total_slices = request->slice_request_size();
  std::vector<std::vector<int>> record(total_node_num);
  std::vector<Request*> node_req_vec(total_node_num);
  for (int i = 0; i < total_node_num; i++) {
    node_req_vec[i] = new Request;
  }

  for (int i = 0; i < total_slices; i++) {
    auto& sr = request->slice_request(i);
    int part_id = sr.slice_partition();
    int co_node = gs_ptr_->part_node_map_.at(part_id);
    record[co_node - 1].emplace_back(i);
    if (co_node == gs_ptr_->node_id_) continue;
    auto slice = node_req_vec[co_node - 1]->add_slice_request();
    slice->set_slice_partition(sr.slice_partition());
    slice->set_data_start(sr.data_start());
    slice->set_data_len(sr.data_len());
  }

  std::vector<Response*> node_res_vec(total_node_num, nullptr);
  std::vector<std::promise<bool>> prom_vec(total_node_num);
  std::vector<std::future<bool>> futu_vec(total_node_num);
  for (int nid = 1; nid <= total_node_num; nid++) {
    if (nid == gs_ptr_->node_id_) continue;
    futu_vec[nid - 1] = prom_vec[nid - 1].get_future();
    node_req_vec[nid - 1]->set_model_version(info_ptr->model_name_.c_str());
    gs_ptr_->pool_ctx_.post([this, nid, &node_req_vec, &node_res_vec, &prom_vec] {
      node_res_vec[nid - 1] = gs_ptr_->node_client_map_.at(nid)->FetchRemoteSlice(*node_req_vec[nid - 1]);
      prom_vec[nid - 1].set_value(true);
    });
  }

  response->mutable_slice_data()->Reserve(total_slices);
  for (int i = 0; i < total_slices; i++) response->add_slice_data();

  // for local node
  int local_num = record[gs_ptr_->node_id_ - 1].size();
  for (int i = 0; i < local_num; i++) {
    int idx = record[gs_ptr_->node_id_ - 1][i];
    auto& sr = request->slice_request(idx);
    int part = sr.slice_partition();
    int start = sr.data_start();
    int length = sr.data_len();
    char* addr = info_ptr->part_buf_map_.at(part) + start;
    response->set_slice_data(idx, addr, length);
  }

  for (int nid = 1; nid <= total_node_num; nid++) {
    if (nid == gs_ptr_->node_id_) continue;
    CHECK(futu_vec[nid - 1].get());
    auto node_res = node_res_vec[nid - 1];
    int local_num = record[nid - 1].size();
    CHECK(node_res->slice_data_size() == local_num);
    for (int i = 0; i < local_num; i++) {
      int idx = record[nid - 1][i];
      auto& res = node_res->slice_data(i);
      response->set_slice_data(idx, res.c_str(), res.size());
    }
  }
  response->set_status(0);

  for (int nid = 1; nid <= total_node_num; nid++) {
    if (node_req_vec[nid - 1]) delete node_req_vec[nid - 1];
    if (node_res_vec[nid - 1]) delete node_res_vec[nid - 1];
  }
  return Status::OK;
}

Status InterNodeServiceImpl::Fetch(ServerContext* ctx, const Request* request, Response* response) {
  std::string goal_model = request->model_version();
  ModelSliceInfo* info_ptr = nullptr;
  std::shared_lock<std::shared_mutex> locker(gs_ptr_->model_info_mtx_);
  for (auto& val : gs_ptr_->model_info_vec_) {
    if (val->model_name_ == goal_model) {
      info_ptr = val; break;
    }
  }
  locker.unlock();
  CHECK(info_ptr != nullptr);
  int local_num = request->slice_request_size();
  response->mutable_slice_data()->Reserve(local_num);
  for (int i = 0; i < local_num; i++) {
    auto& sr = request->slice_request(i);
    int part = sr.slice_partition();
    int start = sr.data_start();
    int length = sr.data_len();
    char* addr = info_ptr->part_buf_map_.at(part) + start;
    response->add_slice_data(addr, length);
  }
  response->set_status(0);
  return Status::OK;
}


int main (int argc, char** argv)
{
  CHECK(argc == 2);
  GlobalServer server(atoi(argv[1]));
  server.RunGlobalServer();
  return 0;
}