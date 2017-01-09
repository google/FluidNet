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

struct CudaVec3 {
  constexpr static const float kEpsilon = 1e-6f;

  float x;
  float y;
  float z;

  __host__ __device__ CudaVec3() : x(0), y(0), z(0) { }
  __host__ __device__ CudaVec3(float _x, float _y, float _z) :
      x(_x), y(_y), z(_z) { }

  __host__ __device__ CudaVec3& operator=(const CudaVec3& other) {
    if (this != &other) {
      this->x = other.x;
      this->y = other.y;
      this->z = other.z;
    }
    return *this;
  }

  __host__ __device__ CudaVec3& operator+=(const CudaVec3& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }
 
  __host__ __device__ const CudaVec3 operator+(const CudaVec3& rhs) const {
    CudaVec3 ret = *this;
    ret += rhs;
    return ret;
  }

  __host__ __device__ CudaVec3& operator-=(const CudaVec3& rhs) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this; 
  }
  
  __host__ __device__ const CudaVec3 operator-(const CudaVec3& rhs) const {
    CudaVec3 ret = *this;
    ret -= rhs; 
    return ret;
  }

  __host__ __device__ const CudaVec3 operator+(const float rhs) const {
    CudaVec3 ret = *this;
    ret.x += rhs;
    ret.y += rhs;
    ret.z += rhs;
    return ret;
  }

  __host__ __device__ const CudaVec3 operator-(const float rhs) const {
    CudaVec3 ret = *this;
    ret.x -= rhs;
    ret.y -= rhs;
    ret.z -= rhs;
    return ret;
  }  
  
  __host__ __device__ const CudaVec3 operator*(const float rhs) const {
    CudaVec3 ret = *this;
    ret.x *= rhs;
    ret.y *= rhs;
    ret.z *= rhs;
    return ret;
  }

  __host__ __device__ inline float& operator()(int32_t i) {
    switch (i) {
    case 0:
      return this->x;
    case 1:
      return this->y;
    case 2:
      return this->z;
    default:
#ifndef __CUDACC__
      THError("vec3 out of bounds.");
      exit(-1);
#else
      // Can't return / raise errors on the GPU. CPU test should catch this.
      return this->x;
#endif
    }
  }

  __host__ __device__ inline float operator()(int32_t i) const {
    return (*this)(i);
  }

  __host__ __device__ inline float norm() const {
    const float length_sq =
        this->x * this->x + this->y * this->y + this->z * this->z;
    if (length_sq > static_cast<float>(kEpsilon)) {
      return std::sqrt(length_sq);
    } else {
      return static_cast<float>(0);
    }
  }

  __host__ __device__ inline void normalize() {
    const float norm = this->norm();
    if (norm > static_cast<float>(kEpsilon)) {
      this->x /= norm;
      this->y /= norm;
      this->z /= norm;
    } else {
      this->x = 0;
      this->y = 0;
      this->z = 0;
    }
  }

  __host__ __device__ static CudaVec3 cross(const CudaVec3& a,
                              const CudaVec3& b) {
    CudaVec3 ret;
    ret.x = (a.y * b.z) - (a.z * b.y);
    ret.y = (a.z * b.x) - (a.x * b.z);
    ret.z = (a.x * b.y) - (a.y * b.x);
    return ret;
  }
};

__host__ __device__ Int3 toInt3(const CudaVec3& val) {
  return Int3(static_cast<int32_t>(val.x),
              static_cast<int32_t>(val.y),
              static_cast<int32_t>(val.z));
};
