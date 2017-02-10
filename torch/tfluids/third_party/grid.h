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
// NOTE: THERE ARE NO CUDA IMPLEMENTATIONS OF THESE. You will need to replicate
// functionally any methods here as flat cuda functions.

#include <iostream>
#include <sstream>
#include <mutex>

class tfluids_(GridBase) {
public:
  // Note: tensors (grid) passed to GridBase will remain owned by the caller.
  // The caller is expected to make sure the pointers remain valid while
  // the GridBase instance is used (and to call THTensor_(free) if required).
  // TODO(tompson): This class should really be pure virtual.
  explicit tfluids_(GridBase)(THTensor* grid, bool is_3d);

  int32_t nbatch() const { return tensor_->size[0]; }
  int32_t nchan() const { return tensor_->size[1]; }
  int32_t zsize() const { return tensor_->size[2]; }
  int32_t ysize() const { return tensor_->size[3]; }
  int32_t xsize() const { return tensor_->size[4]; }

  int32_t bstride() const { return tensor_->stride[0]; }
  int32_t cstride() const { return tensor_->stride[1]; }
  int32_t zstride() const { return tensor_->stride[2]; }
  int32_t ystride() const { return tensor_->stride[3]; }
  int32_t xstride() const { return tensor_->stride[4]; }

  bool is_3d() const { return is_3d_; }
  Int3 getSize() const { return Int3(xsize(), ysize(), zsize()); }

  real getDx() const;

  bool isInBounds(const Int3& p, int bnd) const;

  bool isInBounds(const tfluids_(vec3)& p, int bnd) const;

private:
  // Note: Child classes should use getters!
  THTensor* const tensor_;
  real* const p_grid_;  // The actual flat storage.
  const bool is_3d_;
  static std::mutex mutex_;

  // The indices i, j, k, c, b are x, y, z, chan and batch respectively.
  int32_t index5d(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const;

protected:
  // Use operator() methods in child classes to get at data.
  // Note: if the storage is offset (i.e. because we've selected along the
  // batch dim), this is taken care of in THTensor_(data) (i.e. it returns
  // self->storage->data + self->storageOffset).
  // Build index is used in interpol and interpolComponent. It replicates
  real& data(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return p_grid_[index5d(i, j, k, c, b)];
  }

  real data(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return p_grid_[index5d(i, j, k, c, b)];
  }

  real& data(const Int3& pos, int32_t c, int32_t b) {
    return data(pos.x, pos.y, pos.z, c, b);
  }

  real data(const Int3& pos, int32_t c, int32_t b) const {
    return data(pos.x, pos.y, pos.z, c, b);
  }

  // the BUILD_INDEX macro in Manta's util/interpol.h.
  void buildIndex(int32_t& xi, int32_t& yi, int32_t& zi,
                  real& s0, real& t0, real& f0,
                  real& s1, real& t1, real& f1,
                  const tfluids_(vec3)& pos) const;
};

class tfluids_(FlagGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(FlagGrid)(THTensor* grid, bool is_3d);

  real& operator()(int32_t i, int32_t j, int32_t k, int32_t b) {
    return data(i, j, k, 0, b);
  }
  
  real operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  }

  bool isFluid(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeFluid;
  }

  bool isFluid(const Int3& pos, int32_t b) const {
    return isFluid(pos.x, pos.y, pos.z, b);
  }

  bool isFluid(const tfluids_(vec3)& pos, int32_t b) const {
    return isFluid((int32_t)pos.x, (int32_t)pos.y, (int32_t)pos.z, b);
  }

  bool isObstacle(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeObstacle;
  }

  bool isObstacle(const Int3& pos, int32_t b) const {
    return isObstacle(pos.x, pos.y, pos.z, b);
  }

  bool isObstacle(const tfluids_(vec3)& pos, int32_t b) const {
    return isObstacle((int32_t)pos.x, (int32_t)pos.y, (int32_t)pos.z, b);
  }

  bool isStick(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeStick;
  }

  bool isEmpty(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeEmpty;
  }

  bool isOutflow(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeOutflow;
  }

  bool isOutOfDomain(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return (i < 0 || i >= xsize() || j < 0 || j >= ysize() || k < 0 ||
            k >= zsize() || b < 0 || b >= nbatch());
  }
};

// Our RealGrid is supposed to be like Grid<Real> in Manta.
class tfluids_(RealGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(RealGrid)(THTensor* grid, bool is_3d);

  real& operator()(int32_t i, int32_t j, int32_t k, int32_t b) {
    return data(i, j, k, 0, b);
  }

  real operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  };

  real getInterpolatedHi(const tfluids_(vec3)& pos, int32_t order,
                         int32_t b) const;
  real getInterpolatedWithFluidHi(const tfluids_(FlagGrid)& flag,
                                  const tfluids_(vec3)& pos, int32_t order,
                                  int32_t b) const;

  real interpol(const tfluids_(vec3)& pos, int32_t b) const;
  real interpolWithFluid(const tfluids_(FlagGrid)& flag,
                         const tfluids_(vec3)& pos, int32_t b) const;
private:
  // Interpol1DWithFluid will return:
  // 1. is_fluid = false if a and b are not fluid.
  // 2. is_fluid = true and data(a) if b is not fluid.
  // 3. is_fluid = true and data(b) if a is not fluid.
  // 4. The linear interpolation between data(a) and data(b) if both are fluid.
  static void interpol1DWithFluid(const real val_a, const bool is_fluid_a,
                                  const real val_b, const bool is_fluid_b,
                                  const real t_a, const real t_b,
                                  bool* is_fluid_ab, real* val_ab);
};

class tfluids_(MACGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(MACGrid)(THTensor* grid, bool is_3d);

  // Note: as per other functions, we DO NOT bounds check getCentered. You must
  // not call this method on the edge of the simulation domain.
  const tfluids_(vec3) getCentered(int32_t i, int32_t j, int32_t k,
                                   int32_t b) const;
  
  const tfluids_(vec3) getCentered(const tfluids_(vec3) vec, int32_t b) {
    return getCentered((int32_t)vec.x, (int32_t)vec.y, (int32_t)vec.z, b);
  }

  const tfluids_(vec3) operator()(int32_t i, int32_t j,
                                  int32_t k, int32_t b) const {
    tfluids_(vec3) ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? static_cast<real>(0) : data(i, j, k, 2, b);
    return ret;
  }

  real& operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return data(i, j, k, c, b);
  }

  real operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // MACGrid is 2D.
  void setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
               const tfluids_(vec3)& val);

  tfluids_(vec3) getAtMACX(int32_t i, int32_t j, int32_t k, int32_t b) const;
  tfluids_(vec3) getAtMACY(int32_t i, int32_t j, int32_t k, int32_t b) const;
  tfluids_(vec3) getAtMACZ(int32_t i, int32_t j, int32_t k, int32_t b) const;

  real getInterpolatedComponentHi(const tfluids_(vec3)& pos,
                                  int32_t order, int32_t c, int32_t b) const;
private:
  real interpolComponent(const tfluids_(vec3)& pos, int32_t c, int32_t b) const;
};

class tfluids_(VecGrid) : public tfluids_(GridBase) {
public:
  explicit tfluids_(VecGrid)(THTensor* grid, bool is_3d);

  const tfluids_(vec3) operator()(int32_t i, int32_t j, int32_t k,
                                  int32_t b) const {
    tfluids_(vec3) ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? static_cast<real>(0) : data(i, j, k, 2, b);
    return ret;
  }

  real& operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return data(i, j, k, c, b);
  }

  real operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // VecGrid is 2D, but check that it is non-zero.
  void setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
               const tfluids_(vec3)& val);
  // set will ignore the 3rd component of the input vector if the VecGrid is 2D
  // and it will NOT check that the component is non-zero.
  void set(int32_t i, int32_t j, int32_t k, int32_t b,
           const tfluids_(vec3)& val);

  // Note: you CANNOT call curl on the border of the grid (if you do then
  // the data(...) calls will throw an error.
  // Also note that curl in 2D is a scalar, but we will return a vector anyway
  // with the scalar value being in the 3rd dim.
  tfluids_(vec3) curl(int32_t i, int32_t j, int32_t k, int32_t b);
};

