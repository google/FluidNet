// The methods in this file borrow heavily from - or or a direct port of - parts
// of the Mantaflow library. Since the Mantaflow library is GNU GPL V3, we are
// releasing this code under GNU as well as a third_party add on. See
// FluidNet/torch/tfluids/third_party/README for more information.

/******************************************************************************
 *
 * MantaFlow fluid solver framework 
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 ******************************************************************************/

// This is a Cuda version of what we've implemented in generic/grid.cc.
// Note: you should use the factory functions to...Grid() methods to create
// instances of the grid objects. DO NOT CALL THE CLASS CONSTRUCTORS DIRECTLY.

#pragma once

#include <iostream>
#include <sstream>
#include <mutex>

#define CUDART_INF_F  __int_as_float(0x7f800000)
#define CUDART_NAN_F __int_as_float(0x7fffffff)

#include "generic/int3.cu.h"
#include "generic/vec3.cu.h"

static void checkTensor(THCState* state, THCudaTensor* grid, bool is_3d) {
  if (grid->nDimension != 5) {
    THError("toCudaFlagGrid called on a tensor that is not 5D!");
  }
  if (!is_3d && grid->size[2] != 1) {
    THError("2D grids must have unary z-dimension!");
  }
  if (grid->size[1] > 1) {
    // Vector field.
    if (is_3d && !grid->size[1] == 3) {
      THError("3D vector fields must have 3 channels.");
    } else if (!is_3d && grid->size[1] != 2) {
      THError("2D vector fields must have 2 channels.");
    }
  }
  if (!THCudaTensor_isContiguous(state, grid)) {
    THError("Input grid must be contiguous!");
  }
}

class CudaGridBase {
public:
  // TODO(tompson): This calls should really be pure virtual.
  // NOTE: No size checks are done in this constructor (because we can't call
  // THError from __device__. Instead use to...Grid() methods to create
  // instances of the grid from a tensor pointer.
  __host__ __device__ explicit CudaGridBase(
      const THCDeviceTensor<float, 5>& grid, bool is_3d) :
      is_3d_(is_3d), tensor_(grid) { }

  __host__ __device__ __forceinline__ int64_t nbatch() const {
    return tensor_.getSize(0);
  }
  __host__ __device__ __forceinline__ int64_t nchan() const {
    return tensor_.getSize(1);
  }
  __host__ __device__ __forceinline__ int64_t zsize() const {
    return tensor_.getSize(2);
  }
  __host__ __device__ __forceinline__ int64_t ysize() const {
    return tensor_.getSize(3);
  }
  __host__ __device__ __forceinline__ int64_t xsize() const {
    return tensor_.getSize(4);
  }

  __host__ __device__ __forceinline__ int64_t numel() const {
    return xsize() * ysize() * zsize() * nchan() * nbatch();
  }

  // Note: Cuda version doesn't need access to stride. Dereferening is done by
  // the [] operator of THCDeviceTensor accessor classe (THCDeviceSubTensor).

  // Can't call std::max from __device__.
  template <typename T>
  __host__ __device__ __forceinline__ static T max(const T& a, const T& b) {
    return a < b ? b : a;
  }

  __host__ __device__ __forceinline__ float getDx() const {
    
    const int32_t size_max = max(xsize(), max(ysize(), zsize()));
    return 1.0f / static_cast<float>(size_max);
  }

  __host__ __device__ __forceinline__ bool is_3d() const { return is_3d_; }

  __host__ __device__ __forceinline__ Int3 getSize() const {
    return Int3(xsize(), ysize(), zsize());
  }

  __host__ __device__ __forceinline__ bool isInBounds(const Int3& p,
                                                      int bnd) const {
    bool ret = (p.x >= bnd && p.y >= bnd && p.x < xsize() - bnd &&
                p.y < ysize() - bnd);
    if (is_3d_) {
      ret &= (p.z >= bnd && p.z < zsize() - bnd);
    } else {
      ret &= (p.z == 0);
    }
    return ret; 
  }

  __host__ __device__ __forceinline__ bool isInBounds(const CudaVec3& p,
    int bnd) const {
    return isInBounds(toInt3(p), bnd);
  }

private:
  // Note: Child classes should use getters!
  // Also note: The CUDA version does not use flat indices directly but instead
  // uses the THCDeviceTensor [] operators. We also do not do any bounds checks.
  // Therefore all tests should use the CPU and GPU versions.
  THCDeviceTensor<float, 5> tensor_;
  const bool is_3d_;

protected:
  // Use operator() methods in child classes to get at data.
  __host__ __device__ __forceinline__ float& data(int32_t i, int32_t j,
                                                  int32_t k, int32_t c,
                                                  int32_t b) {
    return tensor_[b][c][k][j][i];
  }
  __host__ __device__ __forceinline__ float data(int32_t i, int32_t j,
                                                 int32_t k, int32_t c,
                                                 int32_t b) const {
    return tensor_[b][c][k][j][i];
  }

  __host__ __device__ __forceinline__ float& data(const Int3& pos, int32_t c,
                                                  int32_t b) {
    return tensor_[b][c][pos.z][pos.y][pos.x];
  }
  __host__ __device__ __forceinline__ float data(const Int3& pos, int32_t c,
                                                 int32_t b) const {
    return tensor_[b][c][pos.z][pos.y][pos.x];
  }

  // Build index is used in interpol and interpolComponent. It replicates
  // the BUILD_INDEX macro in Manta's util/interpol.h.
  __host__ __device__ __forceinline__ void buildIndex(
      int32_t& xi, int32_t& yi, int32_t& zi, float& s0, float& t0, float& f0,
      float& s1, float& t1, float& f1, const CudaVec3& pos) const {
    const float px = pos.x - 0.5f;
    const float py = pos.y - 0.5f;
    const float pz = pos.z - 0.5f;
    xi = static_cast<int32_t>(px);
    yi = static_cast<int32_t>(py);
    zi = static_cast<int32_t>(pz);
    s1 = px - static_cast<float>(xi);
    s0 = 1.0f - s1;
    t1 = py - static_cast<float>(yi);
    t0 = 1.0f - t1;
    f1 = pz - static_cast<float>(zi);
    f0 = 1.0f - f1;
    // Clamp to border.
    if (px < 0.0f) {
      xi = 0;
      s0 = 1.0f;
      s1 = 0.0f;
    }
    if (py < 0.0f) {
      yi = 0;
      t0 = 1.0f;
      t1 = 0.0f;
    }
    if (pz < 0.0f) {
      zi = 0;
      f0 = 1.0f;
      f1 = 0.0f;
    }
    if (xi >= xsize() - 1) {
      xi = xsize() - 2;
      s0 = 0.0f;
      s1 = 1.0f;
    }
    if (yi >= ysize() - 1) {
      yi = ysize() - 2;
      t0 = 0.0f;
      t1 = 1.0f;
    }
    if (zsize() > 1) {
      if (zi >= zsize() - 1) {
        zi = zsize() - 2;
        f0 = 0.0f;
        f1 = 1.0f;
      }
    }
  }
};

class CudaFlagGrid : public CudaGridBase {
public:
  __host__ __device__ explicit CudaFlagGrid(
      const THCDeviceTensor<float, 5>& grid, bool is_3d) :
      CudaGridBase(grid, is_3d) { }

  __host__ __device__ __forceinline__ float& operator()(
      int32_t i, int32_t j, int32_t k, int32_t b) {
    return data(i, j, k, 0, b);
  }
  
  __host__ __device__ __forceinline__ float operator()(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  }

  __host__ __device__ __forceinline__ bool isFluid(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeFluid;
  }

  __host__ __device__ __forceinline__ bool isFluid(
      const Int3& pos, int32_t b) const {
    return static_cast<int>(data(pos.x, pos.y, pos.z, 0, b)) & TypeFluid;
  }

  __host__ __device__ __forceinline__ bool isObstacle(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeObstacle;
  }

  __host__ __device__ __forceinline__ bool isObstacle(
      const Int3& pos, int32_t b) const {
    return isObstacle(pos.x, pos.y, pos.z, b);
  }

  __host__ __device__ __forceinline__ bool isStick(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeStick;
  }

  __host__ __device__ __forceinline__ bool isEmpty(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeEmpty;
  }

  __host__ __device__ __forceinline__ bool isOutflow(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeOutflow;
  }

  __host__ __device__ __forceinline__ bool isOutOfDomain(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return (i < 0 || i >= xsize() || j < 0 || j >= ysize() || k < 0 ||
            k >= zsize() || b < 0 || b >= nbatch());
  }
};

// Our RealGrid is supposed to be like Grid<Real> in Manta.
class CudaRealGrid : public CudaGridBase {
public:
  __host__ __device__ explicit CudaRealGrid(
      const THCDeviceTensor<float, 5>& grid, bool is_3d) :
      CudaGridBase(grid, is_3d) { }

  __host__ __device__ __forceinline__ float& operator()(
      int32_t i, int32_t j, int32_t k, int32_t b) {
    return data(i, j, k, 0, b);
  }

  __host__ __device__ __forceinline__ float operator()(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  };

  __host__ __device__ __forceinline__ float getInterpolatedHi(
      const CudaVec3& pos, int32_t order, int32_t b) const {
    switch (order) {
    case 1:
      return interpol(pos, b);
    case 2:
      // Can't return errors from the kernel. CPU tests should catch.
      break;
    default:
      // Can't return errors from the kernel. CPU tests should catch.
      break;
    }
    return 0;
  }

  __host__ __device__ __forceinline__ float getInterpolatedWithFluidHi(
      const CudaFlagGrid& flags, const CudaVec3& pos, int32_t order,
      int32_t b) const {
    switch (order) {
    case 1:
      return interpolWithFluid(flags, pos, b);
    case 2:
      // Can't return errors from the kernel. CPU tests should catch.
      break;
    default:
      // Can't return errors from the kernel. CPU tests should catch.
      break;
    }
    return 0;
  }

  __host__ __device__ __forceinline__ float interpol(
      const CudaVec3& pos, int32_t b) const {
    int32_t xi, yi, zi;
    float s0, t0, f0, s1, t1, f1;
    buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos); 

    if (is_3d()) {
      return ((data(xi, yi, zi, 0, b) * t0 +
               data(xi, yi + 1, zi, 0, b) * t1) * s0 
          + (data(xi + 1, yi, zi, 0, b) * t0 +
             data(xi + 1, yi + 1, zi, 0, b) * t1) * s1) * f0
          + ((data(xi, yi, zi + 1, 0, b) * t0 +
             data(xi, yi + 1, zi + 1, 0, b) * t1) * s0
          + (data(xi + 1, yi, zi + 1, 0, b) * t0 +
             data(xi + 1, yi + 1, zi + 1, 0, b) * t1) * s1) * f1;
    } else {
       return ((data(xi, yi, 0, 0, b) * t0 +
                data(xi, yi + 1, 0, 0, b) * t1) * s0
          + (data(xi + 1, yi, 0, 0, b) * t0 +
             data(xi + 1, yi + 1, 0, 0, b) * t1) * s1);
    }
  }

  __host__ __device__ __forceinline__ float interpolWithFluid(
      const CudaFlagGrid& flags, const CudaVec3& pos, int32_t ibatch) const {
    int32_t xi, yi, zi;
    float s0, t0, f0, s1, t1, f1;
    buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos);

    if (is_3d()) {
      // val_ab = data(xi, yi, zi, 0, b) * t0 +
      //          data(xi, yi + 1, zi, 0, b) * t1
      const Int3 p_a(xi, yi, zi);
      const Int3 p_b(xi, yi + 1, zi);
      bool is_fluid_ab;
      float val_ab;
      interpol1DWithFluid(data(p_a, 0, ibatch), flags.isFluid(p_a, ibatch),
                          data(p_b, 0, ibatch), flags.isFluid(p_b, ibatch),
                          t0, t1, &is_fluid_ab, &val_ab);

      // val_cd = data(xi + 1, yi, zi, 0, b) * t0 +
      //          data(xi + 1, yi + 1, zi, 0, b) * t1
      const Int3 p_c(xi + 1, yi, zi);
      const Int3 p_d(xi + 1, yi + 1, zi);
      bool is_fluid_cd;
      float val_cd;
      interpol1DWithFluid(data(p_c, 0, ibatch), flags.isFluid(p_c, ibatch),
                          data(p_d, 0, ibatch), flags.isFluid(p_d, ibatch),
                          t0, t1, &is_fluid_cd, &val_cd);

      // val_ef = data(xi, yi, zi + 1, 0, b) * t0 +
      //          data(xi, yi + 1, zi + 1, 0, b) * t1
      const Int3 p_e(xi, yi, zi + 1); 
      const Int3 p_f(xi, yi + 1, zi + 1);
      bool is_fluid_ef;
      float val_ef;
      interpol1DWithFluid(data(p_e, 0, ibatch), flags.isFluid(p_e, ibatch),
                          data(p_f, 0, ibatch), flags.isFluid(p_f, ibatch),
                          t0, t1, &is_fluid_ef, &val_ef);

      // val_gh = data(xi + 1, yi, zi + 1, 0, b) * t0 +
      //          data(xi + 1, yi + 1, zi + 1, 0, b) * t1
      const Int3 p_g(xi + 1, yi, zi + 1); 
      const Int3 p_h(xi + 1, yi + 1, zi + 1);
      bool is_fluid_gh;
      float val_gh;
      interpol1DWithFluid(data(p_g, 0, ibatch), flags.isFluid(p_g, ibatch),
                          data(p_h, 0, ibatch), flags.isFluid(p_h, ibatch),
                          t0, t1, &is_fluid_gh, &val_gh);

      // val_abcd = val_ab * s0 + val_cd * s1
      bool is_fluid_abcd;
      float val_abcd;
      interpol1DWithFluid(val_ab, is_fluid_ab, val_cd, is_fluid_cd,
                          s0, s1, &is_fluid_abcd, &val_abcd);

      // val_efgh = val_ef * s0 + val_gh * s1
      bool is_fluid_efgh;
      float val_efgh;
      interpol1DWithFluid(val_ef, is_fluid_ef, val_gh, is_fluid_gh,
                          s0, s1, &is_fluid_efgh, &val_efgh);

      // val = val_abcd * f0 + val_efgh * f1
      bool is_fluid;
      float val;
      interpol1DWithFluid(val_abcd, is_fluid_abcd, val_efgh, is_fluid_efgh,
                          f0, f1, &is_fluid, &val);

      if (!is_fluid) {
        // None of the 8 cells were fluid. Just return the regular interp
        // of all cells.
        return interpol(pos, ibatch);
      } else { 
        return val;
      }
    } else {
      // val_ab = data(xi, yi, 0, 0, b) * t0 +
      //          data(xi, yi + 1, 0, 0, b) * t1
      const Int3 p_a(xi, yi, 0);
      const Int3 p_b(xi, yi + 1, 0);
      bool is_fluid_ab;
      float val_ab;
      interpol1DWithFluid(data(p_a, 0, ibatch), flags.isFluid(p_a, ibatch),
                          data(p_b, 0, ibatch), flags.isFluid(p_b, ibatch),
                          t0, t1, &is_fluid_ab, &val_ab);

      // val_cd = data(xi + 1, yi, 0, 0, b) * t0 +
      //          data(xi + 1, yi + 1, 0, 0, b) * t1
      const Int3 p_c(xi + 1, yi, 0);
      const Int3 p_d(xi + 1, yi + 1, 0);
      bool is_fluid_cd;
      float val_cd;
      interpol1DWithFluid(data(p_c, 0, ibatch), flags.isFluid(p_c, ibatch),
                          data(p_d, 0, ibatch), flags.isFluid(p_d, ibatch),
                          t0, t1, &is_fluid_cd, &val_cd);

      // val = val_ab * s0 + val_cd * s1
      bool is_fluid;
      float val;
      interpol1DWithFluid(val_ab, is_fluid_ab, val_cd, is_fluid_cd,
                          s0, s1, &is_fluid, &val);

      if (!is_fluid) {
        // None of the 4 cells were fluid. Just return the regular interp
        // of all cells.
        return interpol(pos, ibatch);
      } else {
        return val;
      }
    } 
  }

private:
  // Interpol1DWithFluid will return:
  // 1. is_fluid = false if a and b are not fluid.
  // 2. is_fluid = true and data(a) if b is not fluid.
  // 3. is_fluid = true and data(b) if a is not fluid.
  // 4. The linear interpolation between data(a) and data(b) if both are fluid.
  static __host__ __device__ __forceinline__ void interpol1DWithFluid(
      const float val_a, const bool is_fluid_a,
      const float val_b, const bool is_fluid_b,
      const float t_a, const float t_b,
      bool* is_fluid_ab, float* val_ab) {
    if (!is_fluid_a && !is_fluid_b) {
      *val_ab = 0.0f;
      *is_fluid_ab = false;
    } else if (!is_fluid_a) {
      *val_ab = val_b;
      *is_fluid_ab = true;
    } else if (!is_fluid_b) {
      *val_ab = val_a;
      *is_fluid_ab = true;
    } else {
      *val_ab = val_a * t_a + val_b * t_b;
      *is_fluid_ab = true;
    }
  }
};

class CudaMACGrid : public CudaGridBase {
public:
  __host__ __device__ explicit CudaMACGrid(
      const THCDeviceTensor<float, 5>& grid, bool is_3d) :
      CudaGridBase(grid, is_3d) { }

  // Note: as per other functions, we DO NOT bounds check getCentered. You must
  // not call this method on the edge of the simulation domain.
  __host__ __device__ const CudaVec3 getCentered(
    int32_t i, int32_t j, int32_t k, int32_t b) const {  
    const float x = 0.5f * (data(i, j, k, 0, b) +
                            data(i + 1, j, k, 0, b));
    const float y = 0.5f * (data(i, j, k, 1, b) +
                            data(i, j + 1, k, 1, b));
    const float z = !is_3d() ? 0.0f :
        0.5f * (data(i, j, k, 2, b) +
                data(i, j, k + 1, 2, b));
    return CudaVec3(x, y, z);
  }

  __host__ __device__ __forceinline__ const CudaVec3 operator()(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    CudaVec3 ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? 0.0f : data(i, j, k, 2, b);
    return ret;
  }

  __host__ __device__ __forceinline__ float& operator()(
      int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return data(i, j, k, c, b);
  }

  __host__ __device__ __forceinline__ float operator()(
      int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // MACGrid is 2D.
  __host__ __device__ __forceinline__ void setSafe(
      int32_t i, int32_t j, int32_t k, int32_t b, const CudaVec3& val) {
    data(i, j, k, 0, b) = val.x;
    data(i, j, k, 1, b) = val.y;
    if (is_3d()) {
      data(i, j, k, 2, b) = val.z;
    }
  }

  __host__ __device__ __forceinline__ CudaVec3 getAtMACX(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    CudaVec3 v;
    v.x = data(i, j, k, 0, b);
    v.y = 0.25f * (data(i, j, k, 1, b) + data(i - 1, j, k, 1, b) +
                   data(i, j + 1, k, 1, b) + data(i - 1, j + 1, k, 1, b));
    if (is_3d()) {
      v.z = 0.25f* (data(i, j, k, 2, b) + data(i - 1, j, k, 2, b) +
                    data(i, j, k + 1, 2, b) + data(i - 1, j, k + 1, 2, b));
    } else {
      v.z = 0;
    }
    return v;
  }

  __host__ __device__ __forceinline__ CudaVec3 getAtMACY(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    CudaVec3 v;
    v.x = 0.25f * (data(i, j, k, 0, b) + data(i, j - 1, k, 0, b) +
                   data(i + 1, j, k, 0, b) + data(i + 1, j - 1, k, 0, b));
    v.y = data(i, j, k, 1, b);
    if (is_3d()) {
      v.z = 0.25f* (data(i, j, k, 2, b) + data(i, j - 1, k, 2, b) +
                    data(i, j, k + 1, 2, b) + data(i, j - 1, k + 1, 2, b));
    } else { 
      v.z = 0;
    }
    return v;
  }

  __host__ __device__ __forceinline__ CudaVec3 getAtMACZ(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    CudaVec3 v;
    v.x = 0.25f * (data(i, j, k, 0, b) + data(i, j, k - 1, 0, b) +
                   data(i + 1, j, k, 0, b) + data(i + 1, j, k - 1, 0, b));
    v.y = 0.25f * (data(i, j, k, 1, b) + data(i, j, k - 1, 1, b) +
                   data(i, j + 1, k, 1, b) + data(i, j + 1, k - 1, 1, b));
    if (is_3d()) {
      v.z = data(i, j, k, 2, b);
    } else {
      v.z = 0;
    }
    return v;
  }

  template <int comp>
  __host__ __device__ __forceinline__ float getInterpolatedComponentHi(
      const CudaVec3& pos, int32_t order, int32_t b) const {
    switch (order) {
    case 1:
      return interpolComponent<comp>(pos, b);
    case 2:
      // Can't return errors from the kernel. CPU tests should catch.
      break;
    default:
      // Can't return errors from the kernel. CPU tests should catch.
      break;
    }
    return 0;
  }

  template <int c>
  __host__ __device__ __forceinline__ float interpolComponent(
      const CudaVec3& pos, int32_t b) const {
    int32_t xi, yi, zi;
    float s0, t0, f0, s1, t1, f1;
    buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos);

    if (is_3d()) {
      return ((data(xi, yi, zi, c, b) * t0 +
               data(xi, yi + 1, zi, c, b) * t1) * s0
          + (data(xi + 1, yi, zi, c, b) * t0 +
             data(xi + 1, yi + 1, zi, c, b) * t1) * s1) * f0
          + ((data(xi, yi, zi + 1, c, b) * t0 +
              data(xi, yi + 1, zi + 1, c, b) * t1) * s0
          + (data(xi + 1, yi, zi + 1, c, b) * t0 +
             data(xi + 1, yi + 1, zi + 1, c, b) * t1) * s1) * f1;
    } else {
       return ((data(xi, yi, 0, c, b) * t0 +
                data(xi, yi + 1, 0, c, b) * t1) * s0
          + (data(xi + 1, yi, 0, c, b) * t0 +
             data(xi + 1, yi + 1, 0, c, b) * t1) * s1);
    }
  }
};

class CudaVecGrid : public CudaGridBase {
public:
  __host__ __device__ explicit CudaVecGrid(
      const THCDeviceTensor<float, 5>& grid, bool is_3d) :
      CudaGridBase(grid, is_3d) { }

  __host__ __device__ __forceinline__ const CudaVec3 operator()(
      int32_t i, int32_t j, int32_t k, int32_t b) const {
    CudaVec3 ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? 0.0f : data(i, j, k, 2, b);
    return ret;
  }

  __host__ __device__ __forceinline__ float& operator()(
      int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return data(i, j, k, c, b);
  }

  __host__ __device__ __forceinline__ float operator()(
      int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // VecGrid is 2D.
  __host__ __device__ __forceinline__ void setSafe(
    int32_t i, int32_t j, int32_t k, int32_t b, const CudaVec3& val) {
    data(i, j, k, 0, b) = val.x;
    data(i, j, k, 1, b) = val.y;
    if (is_3d()) {
      data(i, j, k, 2, b) = val.z;
    }
  }

  // Note: you CANNOT call curl on the border of the grid (if you do then
  // the data(...) calls will throw an error (in the CPU code).
  // Also note that curl in 2D is a scalar, but we will return a vector anyway
  // with the scalar value being in the 3rd dim.
  __host__ __device__ __forceinline__ CudaVec3 curl(
      int32_t i, int32_t j, int32_t k, int32_t b) {
     CudaVec3 v(0, 0, 0);
     v.z = 0.5f * ((data(i + 1, j, k, 1, b) -
                                      data(i - 1, j, k, 1, b)) -
                                     (data(i, j + 1, k, 0, b) -
                                      data(i, j - 1, k, 0, b)));
    if(is_3d()) {
        v.x = 0.5f * ((data(i, j + 1, k, 2, b) -
                                         data(i, j - 1, k, 2, b)) -
                                        (data(i, j, k + 1, 1, b) -
                                         data(i, j, k - 1, 1, b)));
        v.y = 0.5f * ((data(i, j, k + 1, 0, b) -
                                         data(i, j, k - 1, 0, b)) -
                                        (data(i + 1, j, k, 2, b) -
                                         data(i - 1, j, k, 2, b)));
    }
    return v;
  }
};

// You should ALWAYS use this method to create a new CudaFlagGrid object (on
// the __host__).
static CudaFlagGrid toCudaFlagGrid(THCState* state, THCudaTensor* grid,
                                   bool is_3d) {
  checkTensor(state, grid, is_3d);
  return CudaFlagGrid(toDeviceTensor<float, 5>(state, grid), is_3d);
}

// You should ALWAYS use this method to create a new CudaFlagGrid object (on
// the __host__).
static CudaRealGrid toCudaRealGrid(THCState* state, THCudaTensor* grid,
                                   bool is_3d) {
  checkTensor(state, grid, is_3d);
  return CudaRealGrid(toDeviceTensor<float, 5>(state, grid), is_3d);
}

// You should ALWAYS use this method to create a new CudaFlagGrid object (on
// the __host__).
static CudaMACGrid toCudaMACGrid(THCState* state, THCudaTensor* grid,
                                 bool is_3d) {
  checkTensor(state, grid, is_3d);
  return CudaMACGrid(toDeviceTensor<float, 5>(state, grid), is_3d);
}

// You should ALWAYS use this method to create a new CudaFlagGrid object (on
// the __host__).
static CudaVecGrid toCudaVecGrid(THCState* state, THCudaTensor* grid,
                                 bool is_3d) {
  checkTensor(state, grid, is_3d);
  return CudaVecGrid(toDeviceTensor<float, 5>(state, grid), is_3d);
}
