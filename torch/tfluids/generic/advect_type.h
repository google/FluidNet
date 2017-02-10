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

#pragma once

#include <luaT.h>
#include <string>

typedef enum {
  ADVECT_EULER_MANTA = 0,  // Port of Manta's implementation.
  ADVECT_MACCORMACK_MANTA = 1,  // Port of Manta's implementation.
  ADVECT_EULER_OURS = 2,
  ADVECT_RK2_OURS = 3,
  ADVECT_RK3_OURS = 4,
  ADVECT_MACCORMACK_OURS = 5,
} AdvectMethod;

AdvectMethod StringToAdvectMethod(lua_State* L, const std::string& str);
