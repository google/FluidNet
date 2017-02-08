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

#include <sstream>
#include "generic/advect_type.h"

AdvectMethod StringToAdvectMethod(lua_State* L, const std::string& str) {
  if (str == "euler") {
    return ADVECT_EULER_MANTA;
  } else if (str == "maccormack") {
    return ADVECT_MACCORMACK_MANTA;
  } else if (str == "rk2Ours") {
    return ADVECT_RK2_OURS;
  } else if (str == "eulerOurs") {
    return ADVECT_EULER_OURS;
  } else if (str == "rk3Ours") {
    return ADVECT_RK3_OURS;
  } else if (str == "maccormackOurs") {
    return ADVECT_MACCORMACK_OURS;
  } else {
    std::stringstream ss;
    ss << "advection method (" << str << ") not supported (options "
       << "are: euler, maccormack, rk2Ours, rk3Ours, eulerOurs)";
    luaL_error(L, ss.str().c_str());
  }
}

