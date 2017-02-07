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

// This is a very, very barebones replication of the Manta grids. It's just
// so that we can more easily transfer KERNEL functions across.

#include <iostream>
#include <sstream>
#include <mutex>

#include "third_party/grid.h"

tfluids_(GridBase)::tfluids_(GridBase)(THTensor* grid, bool is_3d) :
     is_3d_(is_3d), tensor_(grid), p_grid_(THTensor_(data)(grid)) {
  if (grid->nDimension != 5) {
    THError("GridBase: dim must be 5D (even if simulation is 2D).");
  }

  if (!is_3d_ && zsize() != 1) {
    THError("GridBase: 2D grid must have zsize == 1.");
  }
}

real tfluids_(GridBase)::getDx() const {
  const int32_t size_max = std::max(xsize(), std::max(ysize(), zsize()));
  return static_cast<real>(1) / static_cast<real>(size_max);
}

bool tfluids_(GridBase)::isInBounds(const Int3& p, int bnd) const {
  bool ret = (p.x >= bnd && p.y >= bnd && p.x < xsize() - bnd &&
              p.y < ysize() - bnd);
  if (is_3d_) {
    ret &= (p.z >= bnd && p.z < zsize() - bnd);
  } else {
    ret &= (p.z == 0);
  }
  return ret; 
}

bool tfluids_(GridBase)::isInBounds(const tfluids_(vec3)& p,
                                    int bnd) const {
  return isInBounds(toInt3(p), bnd);
}

int32_t tfluids_(GridBase)::index5d(int32_t i, int32_t j, int32_t k,
                                    int32_t c, int32_t b) const {
  if (i >= xsize() || j >= ysize() || k >= zsize() || c >= nchan() ||
      b >= nbatch() || i < 0 || j < 0 || k < 0 || c < 0 || b < 0) {
    std::cout << "Error index5D out of bounds" << std::endl << std::flush;
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "GridBase: index4d out of bounds:" << std::endl
       << "  (i, j, k, c, b) = (" << i << ", " << j
       << ", " << k << ", " << c << ", " << b << "), size = (" << xsize()
       << ", " << ysize() << ", " << zsize() << ", " << nchan() 
       << nbatch() << ")";
    std::cerr << ss.str() << std::endl << "Stack trace:" << std::endl;
    PrintStacktrace();
    std::cerr << std::endl;
    THError("GridBase: index4d out of bounds");
    return 0;
  }
  return (i * xstride() + j * ystride() + k * zstride() + c * cstride() +
          b * bstride());
}

// Build index is used in interpol and interpolComponent. It replicates
// the BUILD_INDEX macro in Manta's util/interpol.h.
void tfluids_(GridBase)::buildIndex(
    int32_t& xi, int32_t& yi, int32_t& zi, real& s0, real& t0, real& f0,
    real& s1, real& t1, real& f1, const tfluids_(vec3)& pos) const {
  const real px = pos.x - static_cast<real>(0.5);
  const real py = pos.y - static_cast<real>(0.5);
  const real pz = pos.z - static_cast<real>(0.5);
  xi = static_cast<int32_t>(px);
  yi = static_cast<int32_t>(py);
  zi = static_cast<int32_t>(pz);
  s1 = px - static_cast<real>(xi);
  s0 = static_cast<real>(1) - s1;
  t1 = py - static_cast<real>(yi);
  t0 = static_cast<real>(1) - t1;
  f1 = pz - static_cast<real>(zi);
  f0 = static_cast<real>(1) - f1;
  // Clamp to border.
  if (px < static_cast<real>(0)) {
    xi = 0;
    s0 = static_cast<real>(1);
    s1 = static_cast<real>(0);
  }
  if (py < static_cast<real>(0)) {
    yi = 0;
    t0 = static_cast<real>(1);
    t1 = static_cast<real>(0);
  }
  if (pz < static_cast<real>(0)) {
    zi = 0;
    f0 = static_cast<real>(1);
    f1 = static_cast<real>(0);
  }
  if (xi >= xsize() - 1) {
    xi = xsize() - 2;
    s0 = static_cast<real>(0);
    s1 = static_cast<real>(1);
  }
  if (yi >= ysize() - 1) {
    yi = ysize() - 2;
    t0 = static_cast<real>(0);
    t1 = static_cast<real>(1);
  }
  if (zsize() > 1) {
    if (zi >= zsize() - 1) {
      zi = zsize() - 2;
      f0 = static_cast<real>(0);
      f1 = static_cast<real>(1);
    }
  }
}

std::mutex tfluids_(GridBase)::mutex_;

tfluids_(FlagGrid)::tfluids_(FlagGrid)(THTensor* grid, bool is_3d) :
    tfluids_(GridBase)(grid, is_3d) {
  if (nchan() != 1) {
    THError("FlagGrid: nchan must be 1 (scalar).");
  }
}

tfluids_(RealGrid)::tfluids_(RealGrid)(THTensor* grid, bool is_3d) :
    tfluids_(GridBase)(grid, is_3d) {
  if (nchan() != 1) {
    THError("RealGrid: nchan must be 1 (scalar).");
  }
}


real tfluids_(RealGrid)::getInterpolatedHi(const tfluids_(vec3)& pos,
                                           int32_t order, int32_t b) const {
  switch (order) {
  case 1:
    return interpol(pos, b);
  case 2:
    THError("getInterpolatedHi ERROR: cubic not supported.");
    // TODO(tompson): implement this.
    break;
  default:
    THError("getInterpolatedHi ERROR: order not supported.");
    break;
  }
  return 0;
}

real tfluids_(RealGrid)::interpol(const tfluids_(vec3)& pos, int32_t b) const {
  int32_t xi, yi, zi;
  real s0, t0, f0, s1, t1, f1;
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

tfluids_(MACGrid)::tfluids_(MACGrid)(THTensor* grid, bool is_3d) :
    tfluids_(GridBase)(grid, is_3d) {
  if (nchan() != 2 && nchan() != 3) {
    THError("MACGrid: input tensor size[0] is not 2 or 3");
  }
  if (!is_3d && zsize() != 1) {
    THError("MACGrid: 2D tensor does not have zsize == 1");
  }
}

// Note: as per other functions, we DO NOT bounds check getCentered. You must
// not call this method on the edge of the simulation domain.
const tfluids_(vec3) tfluids_(MACGrid)::getCentered(
    int32_t i, int32_t j, int32_t k, int32_t b) const {  
  const real x = static_cast<real>(0.5) * (data(i, j, k, 0, b) +
                                           data(i + 1, j, k, 0, b));
  const real y = static_cast<real>(0.5) * (data(i, j, k, 1, b) +
                                           data(i, j + 1, k, 1, b));
  const real z = !is_3d() ? static_cast<real>(0) :
      static_cast<real>(0.5) * (data(i, j, k, 2, b) +
                                data(i, j, k + 1, 2, b));
  return tfluids_(vec3)(x, y, z);
}

// setSafe will ignore the 3rd component of the input vector if the
// MACGrid is 2D.
void tfluids_(MACGrid)::setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                                const tfluids_(vec3)& val) {
  data(i, j, k, 0, b) = val.x;
  data(i, j, k, 1, b) = val.y;
  if (is_3d()) {
    data(i, j, k, 2, b) = val.z;
  } else {
    // This is a pedantic sanity check. We shouldn't be trying to set the
    // z component on a 2D MAC Grid with anything but zero. This is to make
    // sure that the end user fully understands what this function does.
    if (val.z != 0) {
      THError("MACGrid: setSafe z-component is non-zero for a 2D grid.");
    }
  }
}

tfluids_(vec3) tfluids_(MACGrid)::getAtMACX(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  tfluids_(vec3) v;
  v.x = data(i, j, k, 0, b);
  v.y = (real)0.25 * (data(i, j, k, 1, b) + data(i - 1, j, k, 1, b) +
                      data(i, j + 1, k, 1, b) + data(i - 1, j + 1, k, 1, b));
  if (is_3d()) {
    v.z = (real)0.25* (data(i, j, k, 2, b) + data(i - 1, j, k, 2, b) +
                       data(i, j, k + 1, 2, b) + data(i - 1, j, k + 1, 2, b));
  } else {
    v.z = (real)0;
  }
  return v;
}

tfluids_(vec3) tfluids_(MACGrid)::getAtMACY(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  tfluids_(vec3) v;
  v.x = (real)0.25 * (data(i, j, k, 0, b) + data(i, j - 1, k, 0, b) +
                      data(i + 1, j, k, 0, b) + data(i + 1, j - 1, k, 0, b));
  v.y = data(i, j, k, 1, b);
  if (is_3d()) {
    v.z = (real)0.25* (data(i, j, k, 2, b) + data(i, j - 1, k, 2, b) +
                       data(i, j, k + 1, 2, b) + data(i, j - 1, k + 1, 2, b));
  } else { 
    v.z = (real)0;
  }
  return v;
}

tfluids_(vec3) tfluids_(MACGrid)::getAtMACZ(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  tfluids_(vec3) v;
  v.x = (real)0.25 * (data(i, j, k, 0, b) + data(i, j, k - 1, 0, b) +
                      data(i + 1, j, k, 0, b) + data(i + 1, j, k - 1, 0, b));
  v.y = (real)0.25 * (data(i, j, k, 1, b) + data(i, j, k - 1, 1, b) +
                      data(i, j + 1, k, 1, b) + data(i, j + 1, k - 1, 1, b));
  if (is_3d()) {
    v.z = data(i, j, k, 2, b);
  } else {
    v.z = (real)0;
  }
  return v;
}

real tfluids_(MACGrid)::getInterpolatedComponentHi(
    const tfluids_(vec3)& pos, int32_t order, int32_t c, int32_t b) const {
  switch (order) {
  case 1:
    return interpolComponent(pos, c, b);
  case 2:
    THError("getInterpolatedComponentHi ERROR: cubic not supported.");
    // TODO(tompson): implement this.
    break;
  default:
    THError("getInterpolatedComponentHi ERROR: order not supported.");
    break;
  }
  return 0;
}

real tfluids_(MACGrid)::interpolComponent(
    const tfluids_(vec3)& pos, int32_t c, int32_t b) const {
  int32_t xi, yi, zi;
  real s0, t0, f0, s1, t1, f1;
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

tfluids_(VecGrid)::tfluids_(VecGrid)(THTensor* grid, bool is_3d) :
    tfluids_(GridBase)(grid, is_3d) {
  if (nchan() != 2 && nchan() != 3) {
    THError("VecGrid: input tensor size[0] is not 2 or 3");
  }
  if (!is_3d && zsize() != 1) {
    THError("VecGrid: 2D tensor does not have zsize == 1");
  }
}

// setSafe will ignore the 3rd component of the input vector if the
// VecGrid is 2D.
void tfluids_(VecGrid)::setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                                const tfluids_(vec3)& val) {
  data(i, j, k, 0, b) = val.x;
  data(i, j, k, 1, b) = val.y;
  if (is_3d()) {
    data(i, j, k, 2, b) = val.z;
  } else {
    // This is a pedantic sanity check. We shouldn't be trying to set the
    // z component on a 2D Vec Grid with anything but zero. This is to make
    // sure that the end user fully understands what this function does.
    if (val.z != 0) {
      THError("VecGrid: setSafe z-component is non-zero for a 2D grid.");
    }
  }
}

// Note: you CANNOT call curl on the border of the grid (if you do then
// the data(...) calls will throw an error.
// Also note that curl in 2D is a scalar, but we will return a vector anyway
// with the scalar value being in the 3rd dim.
tfluids_(vec3) tfluids_(VecGrid)::curl(int32_t i, int32_t j, int32_t k,
                                       int32_t b) {
   tfluids_(vec3) v(0, 0, 0);
   v.z = static_cast<real>(0.5) * ((data(i + 1, j, k, 1, b) -
                                    data(i - 1, j, k, 1, b)) -
                                   (data(i, j + 1, k, 0, b) -
                                    data(i, j - 1, k, 0, b)));
  if(is_3d()) {
      v.x = static_cast<real>(0.5) * ((data(i, j + 1, k, 2, b) -
                                       data(i, j - 1, k, 2, b)) -
                                      (data(i, j, k + 1, 1, b) -
                                       data(i, j, k - 1, 1, b)));
      v.y = static_cast<real>(0.5) * ((data(i, j, k + 1, 0, b) -
                                       data(i, j, k - 1, 0, b)) -
                                      (data(i + 1, j, k, 2, b) -
                                       data(i - 1, j, k, 2, b)));
  }
  return v;
}

