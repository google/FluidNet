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

#include <TH.h>  // Includes THTensor macro (form THTensor.h).
#include <luaT.h>
#include "third_party/cell_type.h"
#ifndef BUILD_WITHOUT_CUDA_FUNCS
  #include "generic/tfluids.cu.h"
#endif
#include "generic/stack_trace.cc"

// This type is common to both float and double implementations and so has
// to be defined outside tfluids.cc.
#include "generic/int3.cu.h"
#include "generic/advect_type.h"

// Some common functions
inline int32_t clamp(const int32_t x, const int32_t low, const int32_t high) {
  return std::max<int32_t>(std::min<int32_t>(x, high), low);
}

// Expand the CPU types (float and double).  This actually instantiates the
// functions. Note: the order here is important.

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define tfluids_(NAME) TH_CONCAT_3(tfluids_, Real, NAME)

#define real float
#define accreal double
#define Real Float
#define THInf FLT_MAX
#define TH_REAL_IS_FLOAT
#include "generic/vec3.cc"
#include "third_party/grid.cc"
#include "generic/find_connected_fluid_components.cc"
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
#include "generic/vec3.cc"
#include "third_party/grid.cc"
#include "generic/find_connected_fluid_components.cc"
#include "generic/tfluids.cc"
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_DOUBLE

// setfieldint pushes the index and the field value on the stack, and then
// calls lua_settable. The setfield function assumes that before the call the
// table is at the top of the stack (index -1).
void setfieldint(lua_State* L, const char* index, int value) {
  lua_pushstring(L, index);
  lua_pushnumber(L, value);
  lua_settable(L, -3);
}

LUA_EXTERNC DLL_EXPORT int luaopen_libtfluids(lua_State *L) {
  tfluids_FloatMain_init(L);
  tfluids_DoubleMain_init(L);
#ifndef BUILD_WITHOUT_CUDA_FUNCS
  tfluids_CudaMain_init(L);
#endif

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "tfluids");

  lua_newtable(L);
  luaT_setfuncs(L, tfluids_DoubleMain__, 0);
  lua_setfield(L, -2, "double");

  lua_newtable(L);
  luaT_setfuncs(L, tfluids_FloatMain__, 0);
  lua_setfield(L, -2, "float");

#ifndef BUILD_WITHOUT_CUDA_FUNCS
  lua_newtable(L);
  luaT_setfuncs(L, tfluids_CudaMain_getMethodsTable(), 0);
  lua_setfield(L, -2, "cuda");
#endif

  // Create the CellType enum table.
  lua_newtable(L);
  setfieldint(L, "TypeNone", TypeNone);
  setfieldint(L, "TypeFluid", TypeFluid);
  setfieldint(L, "TypeObstacle", TypeObstacle);
  setfieldint(L, "TypeEmpty", TypeEmpty);
  setfieldint(L, "TypeInflow", TypeInflow);
  setfieldint(L, "TypeOutflow", TypeOutflow);
  setfieldint(L, "TypeOpen", TypeOpen);
  setfieldint(L, "TypeStick", TypeStick);
  setfieldint(L, "TypeReserved", TypeReserved);
  setfieldint(L, "TypeZeroPressure", TypeZeroPressure);
  lua_setfield(L, -2, "CellType");

#ifndef BUILD_WITHOUT_CUDA_FUNCS
  lua_pushboolean(L, true);
#else
  std::cout << "WARNING: tfluids compiled without CUDA." << std::endl;
  lua_pushboolean(L, false);
#endif
  lua_setfield(L, -2, "withCUDA");

  return 1;
}
