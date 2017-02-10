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

#include <cmath>
#include <limits>

struct tfluids_(vec3) {
#ifdef TH_REAL_IS_FLOAT
  constexpr static const real kEpsilon = 1e-6f;
#endif
#ifdef TH_REAL_IS_DOUBLE
  constexpr static const real kEpsilon = 1e-10;
#endif

  real x;
  real y;
  real z;

  tfluids_(vec3)() : x(0), y(0), z(0) { }
  tfluids_(vec3)(real _x, real _y, real _z) : x(_x), y(_y), z(_z) { }

  tfluids_(vec3)& operator=(const tfluids_(vec3)& other) {  // Copy assignment.
    if (this != &other) {
      this->x = other.x;
      this->y = other.y;
      this->z = other.z;
    }
    return *this;
  }

  tfluids_(vec3)& operator+=(const tfluids_(vec3)& rhs) {  // accum vec
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }
 
  const tfluids_(vec3) operator+(const tfluids_(vec3)& rhs) const {  // add vec
    tfluids_(vec3) ret = *this;
    ret += rhs;
    return ret;
  }

  tfluids_(vec3)& operator-=(const tfluids_(vec3)& rhs) {  // neg accum vec
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this; 
  }
  
  const tfluids_(vec3) operator-(const tfluids_(vec3)& rhs) const {  // sub vec
    tfluids_(vec3) ret = *this;
    ret -= rhs; 
    return ret;
  }

  const tfluids_(vec3) operator+(const real rhs) const {  // add scalar
    tfluids_(vec3) ret = *this;
    ret.x += rhs;
    ret.y += rhs;
    ret.z += rhs;
    return ret;
  }

  const tfluids_(vec3) operator-(const real rhs) const {  // sub scalar
    tfluids_(vec3) ret = *this;
    ret.x -= rhs;
    ret.y -= rhs;
    ret.z -= rhs;
    return ret;
  }  
  
  const tfluids_(vec3) operator*(const real rhs) const {  // mult scalar
    tfluids_(vec3) ret = *this;
    ret.x *= rhs;
    ret.y *= rhs;
    ret.z *= rhs;
    return ret;
  }

  const tfluids_(vec3) operator/(const real rhs) const {  // mult scalar
    tfluids_(vec3) ret = *this;
    ret.x /= rhs;
    ret.y /= rhs;
    ret.z /= rhs;
    return ret;
  }

  inline real& operator()(int32_t i) {
    switch (i) {
    case 0:
      return this->x;
    case 1:
      return this->y;
    case 2:
      return this->z;
    default:
      THError("vec3 out of bounds.");
      exit(-1);
      break;
    }
  }

  inline real operator()(int32_t i) const {
    return (*this)(i);
  }

  inline real norm() const {
    const real length_sq =
        this->x * this->x + this->y * this->y + this->z * this->z;
    if (length_sq > static_cast<real>(kEpsilon)) {
      return std::sqrt(length_sq);
    } else {
      return static_cast<real>(0);
    }
  }

  inline void normalize() {
    const real norm = this->norm();
    if (norm > static_cast<real>(kEpsilon)) {
      this->x /= norm;
      this->y /= norm;
      this->z /= norm;
    } else {
      this->x = 0;
      this->y = 0;
      this->z = 0;
    }
  }

  static tfluids_(vec3) cross(const tfluids_(vec3)& a,
                              const tfluids_(vec3)& b) {
    tfluids_(vec3) ret;
    ret.x = (a.y * b.z) - (a.z * b.y);
    ret.y = (a.z * b.x) - (a.x * b.z);
    ret.z = (a.x * b.y) - (a.y * b.x);
    return ret;
  }
};

static Int3 toInt3(const tfluids_(vec3)& val);
