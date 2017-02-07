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

#include <cmath>
#include <limits>

struct Int3 {
  int32_t x;
  int32_t y;
  int32_t z;

  __host__ __device__ Int3() : x(0), y(0), z(0) { }
  __host__ __device__ Int3(int32_t _x, int32_t _y, int32_t _z) :
      x(_x), y(_y), z(_z) { }

  __host__ __device__ Int3& operator=(const Int3& other) {
    if (this != &other) {
      this->x = other.x;
      this->y = other.y;
      this->z = other.z;
    }
    return *this;
  }

  __host__ __device__ Int3& operator+=(const Int3& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }
 
  __host__ __device__ const Int3 operator+(const Int3& rhs) const {
    Int3 ret = *this;
    ret += rhs;
    return ret;
  }

  __host__ __device__ Int3& operator-=(const Int3& rhs) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this; 
  }
  
  __host__ __device__ const Int3 operator-(const Int3& rhs) const {
    Int3 ret = *this;
    ret -= rhs; 
    return ret;
  }

  __host__ __device__ const Int3 operator+(const int32_t rhs) const {
    Int3 ret = *this;
    ret.x += rhs;
    ret.y += rhs;
    ret.z += rhs;
    return ret;
  }

  __host__ __device__ const Int3 operator-(const int32_t rhs) const {
    Int3 ret = *this;
    ret.x -= rhs;
    ret.y -= rhs;
    ret.z -= rhs; 
    return ret;
  }
  
  __host__ __device__ const Int3 operator*(const int32_t rhs) const {
    Int3 ret = *this;
    ret.x *= rhs;
    ret.y *= rhs;
    ret.z *= rhs;
    return ret;
  }
};

