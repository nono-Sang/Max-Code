CXX = g++
CXXFLAGS = -g -std=c++17

LIBFLAGS += -lpthread -letcd-cpp-api -lprotobuf -lgrpc++ -lgrpc -lz -lcpprest -lssl -lcrypto -lboost_system
LIBFLAGS += -L$(HADOOP_HOME)/lib/native/ -lhdfs
LIBFLAGS += -lmodel_slice_reader
INCFLAGS += -I/usr/local/include/etcd/proto/
INCFLAGS += -I$(HADOOP_HOME)/include/

PROTOC = protoc
PROTOS_PATH = ./
vpath %.proto $(PROTOS_PATH)
GRPC_CPP_PLUGIN = grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH ?= `which $(GRPC_CPP_PLUGIN)`

all: mc_server

mc_server: alimama.pb.o alimama.grpc.pb.o mc_server.o
	$(CXX) $^ -o $@ $(LIBFLAGS) $(INCFLAGS)

.PRECIOUS: %.grpc.pb.cc
%.grpc.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --grpc_out=. --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN_PATH) $<

.PRECIOUS: %.pb.cc
%.pb.cc: %.proto
	$(PROTOC) -I $(PROTOS_PATH) --cpp_out=. $<

mc_server.o: mc_server.cc
	$(CXX) -c $(CXXFLAGS) mc_server.cc -o mc_server.o $(LIBFLAGS) $(INCFLAGS)

clean:
	rm -f *.o *.pb.cc *.pb.h mc_server