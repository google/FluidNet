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

// This is just a hacky way to expose our tfluids_(vec3) to the standalone
// matlab MEX wrapper.

#include <cmath>
#include <limits>

typedef struct tfluids_(vec3) {
  real x;
  real y;
  real z;
} tfluids_(MyArrayType);

static real tfluids_(length3)(const tfluids_(vec3)& v) {
  const real length_sq = v.x * v.x + v.y * v.y + v.z * v.z;
  if (length_sq > static_cast<real>(1e-6)) {
    return std::sqrt(length_sq);
  } else {
    return static_cast<real>(0);
  }
}

static void tfluids_(cross3)(const tfluids_(vec3)& a,
                             const tfluids_(vec3)& b,
                             tfluids_(vec3)* out) {
  out->x = a.y * b.z - a.z * b.y;
  out->y = a.z * b.x - a.x * b.z;
  out->z = a.x * b.y - a.y * b.x;
}

static void tfluids_(safeNormalize3)(tfluids_(vec3) *v) {
  const real l = tfluids_(length3)(*v);
  if (l > std::numeric_limits<real>::epsilon()) {
    const real invL = static_cast<real>(1) / l;
    v->x = v->x * invL;
    v->y = v->y * invL;
    v->z = v->z * invL;
  } else {
    v->x = static_cast<real>(0);
    v->y = static_cast<real>(0);
    v->z = static_cast<real>(0);
  }
}

// c = a - b.
static inline void tfluids_(sub)(const tfluids_(vec3)& a,
                                 const tfluids_(vec3)& b,
                                 tfluids_(vec3)* c) {
  c->x = a.x - b.x;
  c->y = a.y - b.y;
  c->z = a.z - b.z;
}

// ret = a dot b.
static inline real tfluids_(dot)(const tfluids_(vec3)& a,
                                 const tfluids_(vec3)& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
