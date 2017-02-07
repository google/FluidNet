// In our grid classes it would be nice to be able to see the stacktrace if
// we encounter an out of bounds access.

#include <cxxabi.h>
#include <execinfo.h>
#include <iostream>
#include <mutex>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static std::mutex global_mutex;

void PrintStacktrace() {
  std::lock_guard<std::mutex> lock(global_mutex);
  const int max_num_frames = 64;
  void *frames[max_num_frames];
  size_t num_frames = backtrace(frames, max_num_frames);

  // Print raw stack track, note: likely all symbol names will be mangled
  // but I can't figure out how to prevent this on nvcc + openmp (i.e. I can
  // get the first frame (this function) using abi::__cxa_demangle, but every
  // frame after that is hidden.
  backtrace_symbols_fd(frames, num_frames, STDERR_FILENO);
}

