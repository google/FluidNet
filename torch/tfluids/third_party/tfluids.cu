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

// *****************************************************************************
// advectScalar
// *****************************************************************************

__global__ void SemiLagrange(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src, 
    CudaRealGrid dst, const float dt, const bool is_levelset,
    const int32_t order_space, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    dst(i, j, k, b) = 0;
    return;
  }

  CudaVec3 pos = (CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f) -
                  vel.getCentered(i, j, k, b) * dt);
  dst(i, j, k, b) = src.getInterpolatedHi(pos, order_space, b);
}

__global__ void SemiLagrangeLoop(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src,
    CudaRealGrid dst, const float dt, const bool is_levelset,
    const int32_t order_space, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    dst(i, j, k, b) = 0;
    return;
  }

  CudaVec3 pos = (CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f) -
                  vel.getCentered(i, j, k, b) * dt);
  dst(i, j, k, b) = src.getInterpolatedHi(pos, order_space, b);
}

__global__ void MacCormackCorrect(
    CudaFlagGrid flags, CudaRealGrid old, CudaRealGrid fwd,
    CudaRealGrid bwd, CudaRealGrid dst, const float strength,
    const bool is_levelset) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  float val = fwd(i, j, k, b);

  if (flags.isFluid(i, j, k, b)) {
    // Only correct inside fluid region.
    val += strength * 0.5f * (old(i, j, k, b) - bwd(i, j, k, b));
  }

  dst(i, j, k, b) = val;
}

__device__ __forceinline__ void getMinMax(float& minv, float& maxv,
                                          const float& val) {
  if (val < minv) {
    minv = val;
  }
  if (val > maxv) {
    maxv = val;
  }
}

template <typename T>
__device__ __forceinline__ T clamp(const T val, const T vmin, const T vmax) {
  if (val < vmin) {
    return vmin;
  }
  if (val > vmax) {
    return vmax;
  }
  return val; 
}

__device__ __forceinline__ float doClampComponent(
    const Int3& gridSize, float dst, CudaRealGrid orig, const float fwd,
    CudaVec3 pos, CudaVec3 vel, const int32_t b) {
  float minv = CUDART_INF_F;
  float maxv = -CUDART_INF_F;

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp<int32_t>(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp<int32_t>(curr_pos.y, 0, gridSize.y - 1); 
    const int32_t k0 = clamp<int32_t>(curr_pos.z, 0, 
                             (orig.is_3d() ? (gridSize.z - 1) : 1));
    const int32_t i1 = i0 + 1;
    const int32_t j1 = j0 + 1;
    const int32_t k1 = (orig.is_3d() ? (k0 + 1) : k0);
    if (!orig.isInBounds(Int3(i0, j0, k0), 0) ||
        !orig.isInBounds(Int3(i1, j1, k1), 0)) {
      return fwd;
    }

    // find min/max around source pos
    getMinMax(minv, maxv, orig(i0, j0, k0, b));
    getMinMax(minv, maxv, orig(i1, j0, k0, b));
    getMinMax(minv, maxv, orig(i0, j1, k0, b));
    getMinMax(minv, maxv, orig(i1, j1, k0, b));

    if (orig.is_3d()) {
      getMinMax(minv, maxv, orig(i0, j0, k1, b));
      getMinMax(minv, maxv, orig(i1, j0, k1, b));
      getMinMax(minv, maxv, orig(i0, j1, k1, b)); 
      getMinMax(minv, maxv, orig(i1, j1, k1, b));
    }
  }

  return clamp<float>(dst, minv, maxv);
}

__global__ void MacCormackClamp(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid dst,
    CudaRealGrid orig, CudaRealGrid fwd, const float dt, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  }

  Int3 gridUpper = flags.getSize() - 1;
  float dval = dst(i, j, k, b);

  dval = doClampComponent(gridUpper, dval, orig, fwd(i, j, k, b),
                          CudaVec3(i, j, k),
                          vel.getCentered(i, j, k, b) * dt, b);

  // Lookup forward/backward, round to closest NB.
  Int3 pos_fwd = toInt3(CudaVec3(i, j, k) +
                        CudaVec3(0.5f, 0.5f, 0.5f) -
                        vel.getCentered(i, j, k, b) * dt);
  Int3 pos_bwd = toInt3(CudaVec3(i, j, k) +
                        CudaVec3(0.5f, 0.5f, 0.5f) +
                        vel.getCentered(i, j, k, b) * dt);

  // Test if lookups point out of grid or into obstacle (note doClampComponent
  // already checks sides, below is needed for valid flags access).
  if (pos_fwd.x < 0 || pos_fwd.y < 0 || pos_fwd.z < 0 ||
      pos_bwd.x < 0 || pos_bwd.y < 0 || pos_bwd.z < 0 ||
      pos_fwd.x > gridUpper.x || pos_fwd.y > gridUpper.y ||
      ((pos_fwd.z > gridUpper.z) && flags.is_3d()) ||
      pos_bwd.x > gridUpper.x || pos_bwd.y > gridUpper.y ||
      ((pos_bwd.z > gridUpper.z) && flags.is_3d()) ||
      flags.isObstacle(pos_fwd, b) || flags.isObstacle(pos_bwd, b) ) {
    dval = fwd(i, j, k, b);
  }

  dst(i, j, k, b) = dval;
}

static int tfluids_CudaMain_advectScalar(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  float dt = static_cast<float>(lua_tonumber(L, 1));
  THCudaTensor* tensor_s = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_fwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  THCudaTensor* tensor_bwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));
  const std::string method = static_cast<std::string>(lua_tostring(L, 8));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 9));
  THCudaTensor* tensor_s_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 10, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid src = toCudaRealGrid(state, tensor_s, is_3d);
  CudaRealGrid dst = toCudaRealGrid(state, tensor_s_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  CudaRealGrid fwd = toCudaRealGrid(state, tensor_fwd, is_3d);
  CudaRealGrid bwd = toCudaRealGrid(state, tensor_bwd, is_3d);

  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }
  const int32_t order = method == "euler" ? 1 : 2;
  const bool is_levelset = false;  // We never advect them.
  const int32_t order_space = 1;

  // Do the forward step.
  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  if (order == 1) {
    LaunchKernel(L, &SemiLagrange, flags,
                 flags, vel, src, dst, dt, is_levelset, order_space, bnd);
    // We're done. The forward Euler step is already in the output array.
    return 0;
  } else {
    LaunchKernel(L, &SemiLagrange, flags,
                 flags, vel, src, fwd, dt, is_levelset, order_space, bnd);
  }

  // Do the backwards step.
  LaunchKernel(L, &SemiLagrange, flags,
               flags, vel, fwd, bwd, -dt, is_levelset, order_space, bnd);

  // Perform the correction.
  const float strength = 1.0f;
  LaunchKernel(L, &MacCormackCorrect, flags,
               flags, src, fwd, bwd, dst, strength, is_levelset);

  // Perform clamping.
  LaunchKernel(L, &MacCormackClamp, flags,
               flags, vel, dst, src, fwd, dt, bnd);

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// advectVel
// *****************************************************************************
__global__ void SemiLagrangeMAC(
    CudaFlagGrid flags, CudaMACGrid vel, CudaMACGrid src,
    CudaMACGrid dst, const float dt, const int32_t order_space,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    dst.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    return;
  }

  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  const CudaVec3 pos(static_cast<float>(i) + 0.5f,
                     static_cast<float>(j) + 0.5f,
                     static_cast<float>(k) + 0.5f);

  CudaVec3 xpos = pos - vel.getAtMACX(i, j, k, b) * dt;
  const float vx = src.getInterpolatedComponentHi<0>(xpos, order_space, b);

  CudaVec3 ypos = pos - vel.getAtMACY(i, j, k, b) * dt;
  const float vy = src.getInterpolatedComponentHi<1>(ypos, order_space, b);

  float vz;
  if (vel.is_3d()) {
    CudaVec3 zpos = pos - vel.getAtMACZ(i, j, k, b) * dt;
    vz = src.getInterpolatedComponentHi<2>(zpos, order_space, b);
  } else {
    vz = 0;
  }

  dst.setSafe(i, j, k, b, CudaVec3(vx, vy, vz));
}

__global__ void MacCormackCorrectMAC(
    CudaFlagGrid flags, CudaMACGrid old, CudaMACGrid fwd, 
    CudaMACGrid bwd, CudaMACGrid dst, const float strength) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  bool skip[3] = {false, false, false};

  if (!flags.isFluid(i, j, k, b)) {
    skip[0] = true;
    skip[1] = true;
    skip[2] = true;
  }

  // Note: in Manta code there's a isMAC boolean that is always true.
  if ((i > 0) && (!flags.isFluid(i - 1, j, k, b))) {
    skip[0] = true;
  }
  if ((j > 0) && (!flags.isFluid(i, j - 1, k, b))) {
    skip[1] = true;
  }
  if (flags.is_3d()) {
    if ((k > 0) && (!flags.isFluid(i, j, k - 1, b))) {
      skip[2] = true;
    }
  }

  CudaVec3 val(0, 0, 0);
 
  const int32_t nchan = flags.is_3d() ? 3 : 2;
  for (int32_t c = 0; c < nchan; ++c) {
    if (skip[c]) {
      val(c) = fwd(i, j, k, c, b);
    } else {
      // perform actual correction with given strength.
      val(c) = fwd(i, j, k, c, b) + strength * 0.5f * (old(i, j, k, c, b) -
                                                       bwd(i, j, k, c, b));
    }
  }

  dst.setSafe(i, j, k, b, val);
}

template <int32_t c>
__device__ __forceinline__ float doClampComponentMAC(
    const Int3& gridSize, float dst, const CudaMACGrid& orig,
    float fwd, const CudaVec3& pos, const CudaVec3& vel,
    int32_t b) {
  float minv = CUDART_INF_F;
  float maxv = -CUDART_INF_F;

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp<int32_t>(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp<int32_t>(curr_pos.y, 0, gridSize.y - 1);
    const int32_t k0 = clamp<int32_t>(curr_pos.z, 0,
                                      (orig.is_3d() ? (gridSize.z - 1) : 1));
    const int32_t i1 = i0 + 1;
    const int32_t j1 = j0 + 1;
    const int32_t k1 = (orig.is_3d() ? (k0 + 1) : k0);
    if (!orig.isInBounds(Int3(i0, j0, k0), 0) ||
        !orig.isInBounds(Int3(i1, j1, k1), 0)) {
      return fwd;
    }

    // find min/max around source pos
    getMinMax(minv, maxv, orig(i0, j0, k0, c, b));
    getMinMax(minv, maxv, orig(i1, j0, k0, c, b));
    getMinMax(minv, maxv, orig(i0, j1, k0, c, b));
    getMinMax(minv, maxv, orig(i1, j1, k0, c, b));

    if (orig.is_3d()) {
      getMinMax(minv, maxv, orig(i0, j0, k1, c, b));
      getMinMax(minv, maxv, orig(i1, j0, k1, c, b));
      getMinMax(minv, maxv, orig(i0, j1, k1, c, b));
      getMinMax(minv, maxv, orig(i1, j1, k1, c, b));
    }
  }

  return clamp<float>(dst, minv, maxv);
}

__global__ void MacCormackClampMAC(
    CudaFlagGrid flags, CudaMACGrid vel, CudaMACGrid dst,
    CudaMACGrid orig, CudaMACGrid fwd, const float dt, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  }

  CudaVec3 pos(static_cast<float>(i), static_cast<float>(j),
               static_cast<float>(k));
  CudaVec3 dval = dst(i, j, k, b);
  CudaVec3 dfwd = fwd(i, j, k, b);
  Int3 gridUpper = flags.getSize() - 1;

  dval.x = doClampComponentMAC<0>(gridUpper, dval.x, orig, dfwd.x, pos,
                                  vel.getAtMACX(i, j, k, b) * dt, b);
  dval.y = doClampComponentMAC<1>(gridUpper, dval.y, orig, dfwd.y, pos,
                                  vel.getAtMACY(i, j, k, b) * dt, b);
  if (flags.is_3d()) {
    dval.z = doClampComponentMAC<2>(gridUpper, dval.z, orig, dfwd.z, pos,
                                    vel.getAtMACZ(i, j, k, b) * dt, b);
  } else {
    dval.z = 0;
  }

  // Note (from Manta): The MAC version currently does not check whether source 
  // points were inside an obstacle! (unlike centered version) this would need
  // to be done for each face separately to stay symmetric.
  
  dst.setSafe(i, j, k, b, dval);
}

static int tfluids_CudaMain_advectVel(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  const float dt = static_cast<float>(lua_tonumber(L, 1));
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_fwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_bwd = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 6));
  const std::string method = static_cast<std::string>(lua_tostring(L, 7));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 8));
  THCudaTensor* tensor_u_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 9, "torch.CudaTensor"));

  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }

  const int32_t order = method == "euler" ? 1 : 2;
  const int32_t order_space = 1;

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);

  // We always do self-advection, but we could point orig to another tensor.
  CudaMACGrid src = toCudaMACGrid(state, tensor_u, is_3d);
  CudaMACGrid dst = toCudaMACGrid(state, tensor_u_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  CudaMACGrid fwd = toCudaMACGrid(state, tensor_fwd, is_3d);
  CudaMACGrid bwd = toCudaMACGrid(state, tensor_bwd, is_3d);

  // Do the forward step.
  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  if (order == 1) {
    LaunchKernel(L, &SemiLagrangeMAC, flags,
                 flags, vel, src, dst, dt, order_space, bnd);
    // We're done. The forward Euler step is already in the output array.
    return 0;
  } else {
    LaunchKernel(L, &SemiLagrangeMAC, flags,
                 flags, vel, src, fwd, dt, order_space, bnd);
  }

  // Do the backwards step.
  LaunchKernel(L, &SemiLagrangeMAC, flags,
               flags, vel, fwd, bwd, -dt, order_space, bnd);

  // Perform the correction.
  const float strength = 1.0f;
  LaunchKernel(L, &MacCormackCorrectMAC, flags,
               flags, src, fwd, bwd, dst, strength);

  // Perform clamping.
  LaunchKernel(L, &MacCormackClampMAC, flags,
               flags, vel, dst, src, fwd, dt, bnd);

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

__global__ void setWallBcsForward(CudaFlagGrid flags, CudaMACGrid vel) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  const bool cur_fluid = flags.isFluid(i, j, k, b);
  const bool cur_obs = flags.isObstacle(i, j, k, b);
  if (!cur_fluid && !cur_obs) {
    return;
  }
  
  // we use i > 0 instead of bnd=1 to check outer wall
  if (i > 0 && flags.isObstacle(i - 1, j, k, b)) {
    // TODO(tompson): Set to (potentially) non-zero obstacle velocity.
    vel(i, j, k, 0, b) = 0;
  }
  if (i > 0 && cur_obs && flags.isFluid(i - 1, j, k, b)) {
    vel(i, j, k, 0, b) = 0;
  }
  if (j > 0 && flags.isObstacle(i, j - 1, k, b)) {
    vel(i, j, k, 1, b) = 0;
  }
  if (j > 0 && cur_obs && flags.isFluid(i, j - 1, k, b)) {
    vel(i, j, k, 1, b) = 0;
  }
  
  if (k > 0 && flags.isObstacle(i, j, k - 1, b)) {
    vel(i, j, k, 2, b) = 0;
  }
  if (k > 0 && cur_obs && flags.isFluid(i, j, k - 1, b)) {
    vel(i, j, k, 2, b) = 0;
  }
  
  if (cur_fluid) {
    if ((i > 0 && flags.isStick(i - 1, j, k, b)) ||
        (i < flags.xsize() - 1 && flags.isStick(i + 1, j, k, b))) {
      vel(i, j, k, 1, b) = 0;
      if (vel.is_3d()) {
        vel(i, j, k, 2, b) = 0;
      }
    }
    if ((j > 0 && flags.isStick(i, j - 1, k, b)) ||
        (j < flags.ysize() - 1 && flags.isStick(i, j + 1, k, b))) {
      vel(i, j, k, 0, b) = 0;
      if (vel.is_3d()) {
        vel(i, j, k, 2, b) = 0;
      }
    }
    if (vel.is_3d() &&
        ((k > 0 && flags.isStick(i, j, k - 1, b)) ||
         (k < flags.zsize() - 1 && flags.isStick(i, j, k + 1, b)))) {
      vel(i, j, k, 0, b) = 0;
      vel(i, j, k, 1, b) = 0;
    }
  }
}

static int tfluids_CudaMain_setWallBcsForward(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &setWallBcsForward, flags,
               flags, vel);

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

__global__ void velocityDivergenceForward(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid rhs, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
 
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    rhs(i, j, k, b) = 0;
    return;
  }

  if (!flags.isFluid(i, j, k, b)) {
    rhs(i, j, k, b) = 0;
    return;
  }

  // compute divergence 
  // no flag checks: assumes vel at obstacle interfaces is set to zero.
  float div = vel(i, j, k, 0, b) - vel(i + 1, j, k, 0, b) +
              vel(i, j, k, 1, b) - vel(i, j + 1, k, 1, b);
  if (flags.is_3d()) {
    div += (vel(i, j, k, 2, b) - vel(i, j, k + 1, 2, b));
  }
  rhs(i, j, k, b) = div;
}

static int tfluids_CudaMain_velocityDivergenceForward(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_u_div = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid rhs = toCudaRealGrid(state, tensor_u_div, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  LaunchKernel(L, &velocityDivergenceForward, flags,
               flags, vel, rhs, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// velocityUpdateForward
// *****************************************************************************

__global__ void velocityUpdateForward(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid pressure,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta doesn't touch the velocity on the boundaries (i.e.
    // it stays constant).
    return;
  }

  if (flags.isFluid(i, j, k, b)) {
    if (flags.isFluid(i - 1, j, k, b)) {
      vel(i, j, k, 0, b) -= (pressure(i, j, k, b) -
                             pressure(i - 1, j, k, b));
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      vel(i, j, k, 1, b) -= (pressure(i, j, k, b) -
                             pressure(i, j - 1, k, b));
    }
    if (flags.is_3d() && flags.isFluid(i, j, k - 1, b)) {
      vel(i, j, k, 2, b) -= (pressure(i, j, k, b) -
                             pressure(i, j, k - 1, b));
    }

    if (flags.isEmpty(i - 1, j, k, b)) {
      vel(i, j, k, 0, b) -= pressure(i, j, k, b);
    }
    if (flags.isEmpty(i, j - 1, k, b)) {
      vel(i, j, k, 1, b) -= pressure(i, j, k, b);
    }
    if (flags.is_3d() && flags.isEmpty(i, j, k - 1, b)) {
      vel(i, j, k, 2, b) -= pressure(i, j, k, b);
    }
  }
  else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
    // don't change velocities in outflow cells   
    if (flags.isFluid(i - 1, j, k, b)) {
      vel(i, j, k, 0, b) += pressure(i - 1, j, k, b);
    } else {
      vel(i, j, k, 0, b)  = 0.f;
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      vel(i, j, k, 1, b) += pressure(i, j - 1, k, b);
    } else {
      vel(i, j, k, 1, b)  = 0.f;
    }
    if (flags.is_3d()) {
      if (flags.isFluid(i, j, k - 1, b)) {
        vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
      } else {
        vel(i, j, k, 2, b)  = 0.f;
      }
    }
  }
}

static int tfluids_CudaMain_velocityUpdateForward(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid pressure = toCudaRealGrid(state, tensor_p, is_3d);
 
  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &velocityUpdateForward, flags,
               flags, vel, pressure, bnd);
  
  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// addBuoyancy
// *****************************************************************************

__global__ void addBuoyancy(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid factor,
    THCDeviceTensor<float, 1> strength, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  } 

  if (!flags.isFluid(i, j, k, b)) {
    return;
  }
  if (flags.isFluid(i - 1, j, k, b)) {
    vel(i, j, k, 0, b) += (0.5f * strength[0] *
                           (factor(i, j, k, b) + factor(i - 1, j, k, b)));
  }
  if (flags.isFluid(i, j - 1, k, b)) {
    vel(i, j, k, 1, b) += (0.5f * strength[1] *
                           (factor(i, j, k, b) + factor(i, j - 1, k, b)));
  }
  if (flags.is_3d() && flags.isFluid(i, j, k - 1, b)) {
    vel(i, j, k, 2, b) += (0.5f * strength[2] *
                           (factor(i, j, k, b) + factor(i, j, k - 1, b)));
  }

}
static int tfluids_CudaMain_addBuoyancy(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_density = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_gravity = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_strength = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  const float dt = static_cast<float>(lua_tonumber(L, 6));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));

  if (tensor_gravity->nDimension != 1 || tensor_gravity->size[0] != 3) {
    luaL_error(L, "ERROR: gravity must be a 3D vector (even in 2D)");
  }
  if (tensor_strength->nDimension != 1 || tensor_strength->size[0] != 3) {
    luaL_error(L, "ERROR: gravity must be a 3D vector (even in 2D)");
  }

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid factor = toCudaRealGrid(state, tensor_density, is_3d);

  THCudaTensor_copy(state, tensor_strength, tensor_gravity);
  THCudaTensor_mul(state, tensor_strength, tensor_strength,
                   -1.0f * dt / flags.getDx());
  THCDeviceTensor<float, 1> dev_strength =
      toDeviceTensor<float, 1>(state, tensor_strength);

  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &addBuoyancy, flags,
               flags, vel, factor, dev_strength, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// vorticityConfinement
// *****************************************************************************

__global__ void AddForceField(
    CudaFlagGrid flags, CudaMACGrid vel, CudaVecGrid force, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  const bool curFluid = flags.isFluid(i, j, k, b);
  const bool curEmpty = flags.isEmpty(i, j, k, b);
  if (!curFluid && !curEmpty) {
    return;
  }

  if (flags.isFluid(i - 1, j, k, b) || 
      (curFluid && flags.isEmpty(i - 1, j, k, b))) {
    vel(i, j, k, 0, b) += (0.5f *
                        (force(i - 1, j, k, 0, b) + force(i, j, k, 0, b)));
  }

  if (flags.isFluid(i, j - 1, k, b) ||
      (curFluid && flags.isEmpty(i, j - 1, k, b))) {
    vel(i, j, k, 1, b) += (0.5f * 
                        (force(i, j - 1, k, 1, b) + force(i, j, k, 1, b)));
  }

  if (flags.is_3d() && (flags.isFluid(i, j, k - 1, b) ||
      (curFluid && flags.isEmpty(i, j, k - 1, b)))) {
    vel(i, j, k, 2, b) += (0.5f *
                        (force(i, j, k - 1, 2, b) + force(i, j, k, 2, b)));
  }
}

__global__ void GetCentered(CudaFlagGrid flags, CudaMACGrid vel,
                            CudaVecGrid centered, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    centered.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    return;
  }
  centered.setSafe(i, j, k, b, vel.getCentered(i, j, k, b));
}

__global__ void GetCurlAndCurlNorm(
    CudaFlagGrid flags, CudaVecGrid centered, CudaVecGrid curl,
    CudaRealGrid curl_norm, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    curl.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    curl_norm(i, j, k, b) = 0;
    return;
  }
  const CudaVec3 cur_curl(centered.curl(i, j, k, b));
  curl.setSafe(i, j, k, b, cur_curl);
  curl_norm(i, j, k, b) = cur_curl.norm();
}

__global__ void GetVorticityConfinementForce(
    CudaFlagGrid flags, CudaVecGrid curl, CudaRealGrid curl_norm,
    const float strength, CudaVecGrid force, const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd || 
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Don't add force on the boundaries.
    force.setSafe(i, j, k, b, CudaVec3(0, 0, 0));
    return;
  }

  CudaVec3 grad(0, 0, 0);
  grad.x = 0.5f * (curl_norm(i + 1, j, k, b) - curl_norm(i - 1, j, k, b));
  grad.y = 0.5f * (curl_norm(i, j + 1, k, b) - curl_norm(i, j - 1, k, b));
  if (flags.is_3d()) {
    grad.z = 0.5f * (curl_norm(i, j, k + 1, b) - curl_norm(i, j, k - 1, b));
  }
  grad.normalize();
  
  force.setSafe(i, j, k, b, CudaVec3::cross(grad, curl(i, j, k, b)) * strength);
}

static int tfluids_CudaMain_vorticityConfinement(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  const float strength = static_cast<float>(lua_tonumber(L, 3));
  THCudaTensor* tensor_centered = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_curl = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  THCudaTensor* tensor_curl_norm = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));
  THCudaTensor* tensor_force = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 7, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 8));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaVecGrid centered = toCudaVecGrid(state, tensor_centered, is_3d);
  CudaVecGrid curl = toCudaVecGrid(state, tensor_curl, true);  // Always 3D.
  CudaRealGrid curl_norm = toCudaRealGrid(state, tensor_curl_norm, is_3d);
  CudaVecGrid force = toCudaVecGrid(state, tensor_force, is_3d);

  // First calculate the centered velocity.
  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  LaunchKernel(L, &GetCentered, flags,
               flags, vel, centered, bnd);

  // Now calculate the curl and it's (l2) norm (of the centered velocities).
  LaunchKernel(L, &GetCurlAndCurlNorm, flags,
               flags, centered, curl, curl_norm, bnd);

 
  // Now calculate the vorticity confinement force.
  LaunchKernel(L, &GetVorticityConfinementForce, flags,
               flags, curl, curl_norm, strength, force, bnd);

  // Now apply the force.
  LaunchKernel(L, &AddForceField, flags,
               flags, vel, force, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

