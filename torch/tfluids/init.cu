// Copyright 2016 Google Inc, NYU.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>

#include <algorithm>

#include <TH.h>
#include <luaT.h>
#include "generic/tfluids.cu.h"

// This type is common to both float and double implementations and so has
// to be defined outside tfluids.cc.
typedef struct Int3 {
  int32_t x;
  int32_t y;
  int32_t z;
} Int3;

inline int32_t IX(const int32_t i, const int32_t j, const int32_t k,
                  const Int3& dims) {
#if defined(DEBUG)
  assert(i >= 0 && i < dims.x);
  assert(j >= 0 && j < dims.y);
  assert(k >= 0 && k < dims.z);
#endif
  return i + j * dims.x + k * dims.x * dims.y;
}

inline int32_t ClampInt32(const int32_t x, const int32_t low,
                          const int32_t high) {
  return std::max<int32_t>(std::min<int32_t>(x, high), low);
}

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define tfluids_(NAME) TH_CONCAT_3(tfluids_, Real, NAME)

// Note: instead of calling THGenerateFloatTypes.h, we're going to hack into
// the torch build system a little bit. This makes the tfluids library
// compatible with the blaze build system (for reasons that aren't interesting,
// but are very annoying).
#define TH_GENERIC_FILE

#define real float
#define accreal double
#define Real Float
#define THInf FLT_MAX
#define TH_REAL_IS_FLOAT
#include "generic/tfluids.cc"
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define THInf DBL_MAX
#define TH_REAL_IS_DOUBLE
#include "generic/tfluids.cc"
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_DOUBLE

#undef TH_GENERIC_FILE

LUA_EXTERNC DLL_EXPORT int luaopen_libtfluids(lua_State *L) {
  tfluids_FloatMain_init(L);
  tfluids_DoubleMain_init(L);
  tfluids_CudaMain_init(L);

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "tfluids");

  lua_newtable(L);
  luaT_setfuncs(L, tfluids_DoubleMain__, 0);
  lua_setfield(L, -2, "double");

  lua_newtable(L);
  luaT_setfuncs(L, tfluids_FloatMain__, 0);
  lua_setfield(L, -2, "float");

  lua_newtable(L);
  luaT_setfuncs(L, tfluids_CudaMain_getMethodsTable(), 0);
  lua_setfield(L, -2, "cuda");

  return 1;
}
