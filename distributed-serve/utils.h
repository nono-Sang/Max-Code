#pragma once

#include <arpa/inet.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <netdb.h>
#include <unistd.h>

const int max_name_size = 100;

#define LOG_INFO(M, ...) \
  fprintf(stderr, "[INFO] (%s:%d) " M "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_ERR(M, ...)                          \
  fprintf(stderr,                                \
          "[ERROR] (%s:%d: errno: %s) " M "\n",  \
          __FILE__,                              \
          __LINE__,                              \
          errno == 0 ? "None" : strerror(errno), \
          ##__VA_ARGS__)

#define CHECK(COND)                        \
  do {                                     \
    if (!(COND)) {                         \
      LOG_ERR("Check failure: %s", #COND); \
      exit(EXIT_FAILURE);                  \
    }                                      \
  } while (0);

void GetLocalIp(std::string &local_ip) {
  char hostname[max_name_size];
  // success return 0, error return -1
  CHECK(gethostname(hostname, sizeof(hostname)) == 0);
  hostent *host = gethostbyname(hostname);
  CHECK(host != nullptr);
  local_ip = inet_ntoa(*reinterpret_cast<in_addr *>(host->h_addr_list[0]));
}

inline void Command(const std::string &command) {
  system(command.c_str());
  LOG_INFO("command is: %s", command.c_str());
}