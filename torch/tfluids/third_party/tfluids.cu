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

__global__ void SemiLagrangeRK2Ours(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src,
    CudaRealGrid dst, const float dt, const int32_t order_space,
    const int32_t bnd, const bool line_trace, const bool sample_outside_fluid) {
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

  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    dst(i, j, k, b) = src(i, j, k, b);
    return;
  }

  const CudaVec3 pos =
      CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f);
  CudaVec3 displacement = vel.getCentered(i, j, k, b) * (-dt * 0.5f);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  CudaVec3 half_pos;
  const bool hit_bnd_half =
      calcLineTrace(pos, displacement, flags, b, &half_pos, line_trace);

  if (hit_bnd_half) {
    // We hit the boundary, then as per Bridson, we should clamp the backwards
    // trace. Note: if we treated this as a full euler step, we would have hit
    // the same blocker because the line trace is linear.
    if (!sample_outside_fluid) {
      dst(i, j, k, b) =
          src.getInterpolatedWithFluidHi(flags, half_pos, order_space, b);
    } else {
      dst(i, j, k, b) =
          src.getInterpolatedHi(half_pos, order_space, b);
    }
    return;
  }

  // Otherwise, sample the velocity at this half-step location and do another
  // backwards trace.
  displacement.x = vel.getInterpolatedComponentHi<0>(half_pos, order_space, b);
  displacement.y = vel.getInterpolatedComponentHi<1>(half_pos, order_space, b);
  if (flags.is_3d()) {
    displacement.z =
        vel.getInterpolatedComponentHi<2>(half_pos, order_space, b);
  }
  displacement = displacement * (-dt);
  CudaVec3 back_pos;
  calcLineTrace(pos, displacement, flags, b, &back_pos, line_trace);

  // Note: It actually doesn't matter if we hit the boundary on the second
  // trace. We clamp the trace anyway.

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    dst(i, j, k, b) =
        src.getInterpolatedWithFluidHi(flags, back_pos, order_space, b);
  } else {
    dst(i, j, k, b) =
        src.getInterpolatedHi(back_pos, order_space, b);
  }
}

__global__ void SemiLagrangeRK3Ours(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src,
    CudaRealGrid dst, const float dt, const int32_t order_space,
    const int32_t bnd, const bool line_trace, const bool sample_outside_fluid) {
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

  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    dst(i, j, k, b) = src(i, j, k, b);
    return;
  }

  // We're implementing the RK3 from Bridson page 242.
  // k1 = f(q^n)
  const CudaVec3 k1_pos =
      CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f);
  CudaVec3 k1 = vel.getCentered(i, j, k, b);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  // k2 = f(q^n - 1/2 * dt * k1)
  CudaVec3 k2_pos;
  if (calcLineTrace(k1_pos, k1 * (-dt * 0.5f), flags, b, &k2_pos, line_trace)) {
    // If we hit the boundary we'll truncate to an Euler step.
    if (!sample_outside_fluid) {
      dst(i, j, k, b) =
        src.getInterpolatedWithFluidHi(flags, k2_pos, order_space, b);
    } else {
      dst(i, j, k, b) =
        src.getInterpolatedHi(k2_pos, order_space, b);
    }
    return;
  }
  CudaVec3 k2;
  k2.x = vel.getInterpolatedComponentHi<0>(k2_pos, order_space, b);
  k2.y = vel.getInterpolatedComponentHi<1>(k2_pos, order_space, b);
  if (flags.is_3d()) {
    k2.z = vel.getInterpolatedComponentHi<2>(k2_pos, order_space, b);
  }

  // k3 = f(q^n - 3/4 * dt * k2)
  CudaVec3 k3_pos;
  if (calcLineTrace(k1_pos, k2 * (-dt * 0.75f), flags, b, &k3_pos,
                    line_trace)) {
    // If we hit the boundary we'll truncate to the k2 position (euler step).
    if (!sample_outside_fluid) {
      dst(i, j, k, b) =
        src.getInterpolatedWithFluidHi(flags, k2_pos, order_space, b);
    } else {
      dst(i, j, k, b) =
        src.getInterpolatedHi(k2_pos, order_space, b);
    }
    return;
  }
  CudaVec3 k3;
  k3.x = vel.getInterpolatedComponentHi<0>(k3_pos, order_space, b);
  k3.y = vel.getInterpolatedComponentHi<1>(k3_pos, order_space, b);
  if (flags.is_3d()) {
    k3.z = vel.getInterpolatedComponentHi<2>(k3_pos, order_space, b);
  } 

  // Finally calculate the effective velocity and perform a line trace.
  CudaVec3 back_pos;
  CudaVec3 displacement = (k1 * (-dt * (2.0f / 9.0f)) +
                           k2 * (-dt * (3.0f / 9.0f)) +
                           k3 * (-dt * (4.0f / 9.0f)));
  calcLineTrace(k1_pos, displacement, flags, b, &back_pos, line_trace);

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    dst(i, j, k, b) =
        src.getInterpolatedWithFluidHi(flags, back_pos, order_space, b);
  } else {
    dst(i, j, k, b) =
        src.getInterpolatedHi(back_pos, order_space, b);
  }
}

// This is the same kernel as our other Euler kernel, except it saves the
// particle trace position. This is used for our maccormack routine (we'll do
// a local search around these positions in our clamp routine).
__global__ void SemiLagrangeEulerOursSavePos(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src,
    CudaRealGrid dst, const float dt, const int32_t order_space,
    const int32_t bnd, const bool line_trace, const bool sample_outside_fluid,
    CudaVecGrid pos) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border.
    dst(i, j, k, b) = 0;
    pos.setSafe(i, j, k, b, CudaVec3(i, j, k) + 0.5f);
    return;
  }

  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    dst(i, j, k, b) = src(i, j, k, b);
    pos.setSafe(i, j, k, b, CudaVec3(i, j, k) + 0.5f);
    return;
  }

  const CudaVec3 start_pos =
      CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f);
  CudaVec3 displacement = vel.getCentered(i, j, k, b) * (-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  CudaVec3 back_pos;
  calcLineTrace(start_pos, displacement, flags, b, &back_pos, line_trace);
  pos.setSafe(i, j, k, b, back_pos);

  // Sample at this back position.
  if (!sample_outside_fluid) {
    dst(i, j, k, b) =
        src.getInterpolatedWithFluidHi(flags, back_pos, order_space, b);
  } else {
    dst(i, j, k, b) =
        src.getInterpolatedHi(back_pos, order_space, b);
  }
}

__global__ void SemiLagrangeEulerOurs(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid src,
    CudaRealGrid dst, const float dt, const int32_t order_space,
    const int32_t bnd, const bool line_trace, const bool sample_outside_fluid) {
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

  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    dst(i, j, k, b) = src(i, j, k, b);
    return;
  }

  const CudaVec3 pos =
      CudaVec3((float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f);
  CudaVec3 displacement = vel.getCentered(i, j, k, b) * (-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  CudaVec3 back_pos;
  calcLineTrace(pos, displacement, flags, b, &back_pos, line_trace);

  // Sample at this back position.
  if (!sample_outside_fluid) {
    dst(i, j, k, b) =
        src.getInterpolatedWithFluidHi(flags, back_pos, order_space, b);
  } else {
    dst(i, j, k, b) =
        src.getInterpolatedHi(back_pos, order_space, b);
  }
}

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
    // Note: there's a fix here, the Manta code clamps between 0 and 1 if
    // not 3D which is wrong (it should be 0 always).
    const int32_t k0 =
        orig.is_3d() ? clamp<int32_t>(curr_pos.z, 0, (gridSize.z - 1)) : 0;
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

// Our version is a little different. It is a search around a single input
// position for min and max values. If no valid values are found, then
// false is returned (indicating that a clamp shouldn't be performed) otherwise
// true is returned (and the clamp min and max bounds are set).
__device__ __forceinline__ float getClampBounds(
    CudaRealGrid src, CudaVec3 pos, const int32_t b,
    CudaFlagGrid flags, const bool sample_outside_fluid, float* clamp_min,
    float* clamp_max) {
  float minv = CUDART_INF_F;
  float maxv = -CUDART_INF_F;

  // clamp forward lookup to grid 
  const int32_t i0 = clamp<int32_t>(pos.x, 0, flags.xsize() - 1);
  const int32_t j0 = clamp<int32_t>(pos.y, 0, flags.ysize() - 1);
  const int32_t k0 =
    src.is_3d() ? clamp<int32_t>(pos.z, 0, flags.zsize() - 1) : 0;
  // Some modification here. Instead of looking just to the RHS, we will search
  // all neighbors within a region.  This is more expensive but better handles
  // border cases.
  int32_t ncells = 0;
  for (int32_t k = k0 - 1; k <= k0 + 1; k++) {
    for (int32_t j = j0 - 1; j <= j0 + 1; j++) {
      for (int32_t i = i0 - 1; i <= i0 + 1; i++) {
        if (k < 0 || k >= flags.zsize() ||
            j < 0 || j >= flags.ysize() ||
            i < 0 || i >= flags.xsize()) {
          // Outside bounds.
          continue;
        } else if (sample_outside_fluid || flags.isFluid(i, j, k, b)) {
          // Either we don't care about clamping to values inside the fluid, or
          // this is a fluid cell...
          getMinMax(minv, maxv, src(i, j, k, b));
          ncells++;
        }
      }
    }
  }

  if (ncells < 1) {
    // Only a single fluid cell found. Return false to indicate that a clamp
    // shouldn't be performed.
    return false;
  } else {
    *clamp_min = minv;
    *clamp_max = maxv;
    return true;
  }
}

__global__ void MacCormackClampOurs(
    CudaFlagGrid flags, CudaMACGrid vel, CudaRealGrid dst,
    CudaRealGrid src, CudaRealGrid fwd, const float dt, const int32_t bnd,
    CudaVecGrid fwd_pos, CudaVecGrid bwd_pos, const bool sample_outside_fluid) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
 
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  }

  // Calculate the clamp bounds.
  float clamp_min = CUDART_INF_F;
  float clamp_max = -CUDART_INF_F;

  // Calculate the clamp bounds around the forward position.
  CudaVec3 pos = fwd_pos(i, j, k, b);
  const bool do_clamp_fwd = getClampBounds(
      src, pos, b, flags, sample_outside_fluid, &clamp_min, &clamp_max);

  // Calculate the clamp bounds around the backward position. Recall that
  // the bwd value was sampled on the fwd output (so src is replaced with fwd).
  // EDIT(tompson): According to "An unconditionally stable maccormack method"
  // only a forward search is required.
  // pos = bwd_pos(i, j, k, b);
  // const bool do_clamp_bwd = getClampBounds(
  //     fwd, pos, b, flags, sample_outside_fluid, &clamp_min, &clamp_max);

  float dval;
  if (!do_clamp_fwd) {
    // If the cell is surrounded by fluid neighbors either in the fwd or
    // backward directions, then we need to revert to an euler step.
    dval = fwd(i, j, k, b);
  } else {
    // We found valid values with which to clamp the maccormack corrected
    // quantity. Apply this clamp.
    dval = clamp<float>(dst(i, j, k, b), clamp_min, clamp_max);
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
  const std::string method_str = static_cast<std::string>(lua_tostring(L, 8));
  THCudaTensor* tensor_fwd_pos = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 9, "torch.CudaTensor"));
  THCudaTensor* tensor_bwd_pos = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 10, "torch.CudaTensor"));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 11));
  const bool sample_outside_fluid = static_cast<bool>(lua_toboolean(L, 12));
  const float maccormack_strength = static_cast<float>(lua_tonumber(L, 13));
  THCudaTensor* tensor_s_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 14, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);
  CudaRealGrid src = toCudaRealGrid(state, tensor_s, is_3d);
  CudaRealGrid dst = toCudaRealGrid(state, tensor_s_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  CudaRealGrid fwd = toCudaRealGrid(state, tensor_fwd, is_3d);
  CudaRealGrid bwd = toCudaRealGrid(state, tensor_bwd, is_3d);
  CudaVecGrid fwd_pos = toCudaVecGrid(state, tensor_fwd_pos, is_3d);
  CudaVecGrid bwd_pos = toCudaVecGrid(state, tensor_bwd_pos, is_3d);

  AdvectMethod method = StringToAdvectMethod(L, method_str);

  const bool is_levelset = false;  // We never advect them.
  const int32_t order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to our
  // methods only).
  const bool line_trace = true;

  // Do the forward step.
  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  if (method == ADVECT_EULER_MANTA) {
    LaunchKernel(L, &SemiLagrange, flags,
                 flags, vel, src, dst, dt, is_levelset, order_space, bnd);
    // We're done. The forward Euler step is already in the output array.
  } else if (method == ADVECT_RK2_OURS) {
    LaunchKernel(L, &SemiLagrangeRK2Ours, flags,
                 flags, vel, src, dst, dt, order_space, bnd, line_trace,
                 sample_outside_fluid);
  } else if (method == ADVECT_EULER_OURS) {
    LaunchKernel(L, &SemiLagrangeEulerOurs, flags,
                 flags, vel, src, dst, dt, order_space, bnd, line_trace,
                 sample_outside_fluid);
  } else if (method == ADVECT_RK3_OURS) {
    LaunchKernel(L, &SemiLagrangeRK3Ours, flags,
                 flags, vel, src, dst, dt, order_space, bnd, line_trace,
                 sample_outside_fluid);
  } else if (method == ADVECT_MACCORMACK_MANTA ||
             method == ADVECT_MACCORMACK_OURS) {
    // Do the forwards step.
    if (method == ADVECT_MACCORMACK_MANTA) {
      LaunchKernel(L, &SemiLagrange, flags,
                   flags, vel, src, fwd, dt, is_levelset, order_space, bnd);
    } else {
      LaunchKernel(L, &SemiLagrangeEulerOursSavePos, flags,
                   flags, vel, src, fwd, dt, order_space, bnd, line_trace,
                   sample_outside_fluid, fwd_pos);
    }

    // Do the backwards step.
    if (method == ADVECT_MACCORMACK_MANTA) {
      LaunchKernel(L, &SemiLagrange, flags,
                   flags, vel, fwd, bwd, -dt, is_levelset, order_space, bnd);
    } else {
      LaunchKernel(L, &SemiLagrangeEulerOursSavePos, flags,
                   flags, vel, fwd, bwd, -dt, order_space, bnd, line_trace,
                   sample_outside_fluid, bwd_pos);
    }

    // Perform the correction.
    LaunchKernel(L, &MacCormackCorrect, flags,
                 flags, src, fwd, bwd, dst, maccormack_strength, is_levelset);

    // Perform clamping.
    if (method == ADVECT_MACCORMACK_MANTA) {
      LaunchKernel(L, &MacCormackClamp, flags,
                   flags, vel, dst, src, fwd, dt, bnd);
    } else {
      LaunchKernel(L, &MacCormackClampOurs, flags,
                   flags, vel, dst, src, fwd, dt, bnd, fwd_pos, bwd_pos,
                   sample_outside_fluid);
    }
  } else {
     std::stringstream ss;
     ss << "advection method (" << method_str << ") is not supported";
     luaL_error(L, ss.str().c_str());
  }

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// advectVel
// *****************************************************************************

// Take a step along the vel from pos and sample the velocity there.
__device__ __forceinline__ bool SemiLagrangeStepMAC(
    const CudaFlagGrid& flags, const CudaMACGrid& vel, const CudaMACGrid& src,
    const float scale, const bool line_trace, const int32_t order_space,
    const CudaVec3& pos, const int32_t i, const int32_t j, const int32_t k,
    const int32_t b, CudaVec3* val) {
  // TODO(tompson): We really want to clamp to the SMALLEST of the steps in each
  // dimension, however this is OK for now (because doing so would expensive)...
  CudaVec3 xpos;
  bool hitx = calcLineTrace(
      pos, vel.getAtMACX(i, j, k, b) * scale, flags, b, &xpos, line_trace);
  val->x = src.getInterpolatedComponentHi<0>(xpos, order_space, b);

  CudaVec3 ypos;
  bool hity = calcLineTrace(
      pos, vel.getAtMACY(i, j, k, b) * scale, flags, b, &ypos, line_trace);
  val->y = src.getInterpolatedComponentHi<1>(ypos, order_space, b);

  bool hitz;
  if (vel.is_3d()) {
    CudaVec3 zpos;
    hitz = calcLineTrace(
        pos, vel.getAtMACZ(i, j, k, b) * scale, flags, b, &zpos, line_trace);
    val->z = src.getInterpolatedComponentHi<2>(zpos, order_space, b);
  } else {
    val->z = 0;
    hitz = false;
  }

  return hitx || hity || hitz; 
}

__global__ void SemiLagrangeEulerOursMAC(
    CudaFlagGrid flags, CudaMACGrid vel, CudaMACGrid src,
    CudaMACGrid dst, const float dt, const int32_t order_space,
    const int32_t bnd, const bool line_trace) {
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

  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    dst.setSafe(i, j, k, b, src(i, j, k, b));
    return;
  }

  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  const CudaVec3 pos(static_cast<float>(i) + 0.5f,
                     static_cast<float>(j) + 0.5f,
                     static_cast<float>(k) + 0.5f);

  CudaVec3 val;
  SemiLagrangeStepMAC(flags, vel, src, -dt, line_trace, order_space, pos,
                      i, j, k, b, &val);

  dst.setSafe(i, j, k, b, val);
}

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
  const std::string method_str = static_cast<std::string>(lua_tostring(L, 7));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 8));
  const float maccormack_strength = static_cast<float>(lua_tonumber(L, 9));
  THCudaTensor* tensor_u_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 10, "torch.CudaTensor"));

  AdvectMethod method = StringToAdvectMethod(L, method_str);

  // TODO(tompson): Implement RK2 and RK3 methods.
  if (method == ADVECT_RK2_OURS || method == ADVECT_RK3_OURS) {
    // We do not yet have an RK2 or RK3 implementation. Use Maccormack.
    method = ADVECT_MACCORMACK_OURS;
  }

  const int32_t order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to our
  // methods only).
  const bool line_trace = true;  

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);

  // We always do self-advection, but we could point orig to another tensor.
  CudaMACGrid src = toCudaMACGrid(state, tensor_u, is_3d);
  CudaMACGrid dst = toCudaMACGrid(state, tensor_u_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  CudaMACGrid fwd = toCudaMACGrid(state, tensor_fwd, is_3d);
  CudaMACGrid bwd = toCudaMACGrid(state, tensor_bwd, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  if (method == ADVECT_EULER_MANTA) {
    LaunchKernel(L, &SemiLagrangeMAC, flags,
                 flags, vel, src, dst, dt, order_space, bnd);
  } else if (method == ADVECT_EULER_OURS) {
    LaunchKernel(L, &SemiLagrangeEulerOursMAC, flags,
                 flags, vel, src, dst, dt, order_space, bnd, line_trace);
  } else if (method == ADVECT_MACCORMACK_MANTA ||
             method == ADVECT_MACCORMACK_OURS) {
    // Do the forward step.
    if (method == ADVECT_MACCORMACK_MANTA) {
      LaunchKernel(L, &SemiLagrangeMAC, flags,
                   flags, vel, src, fwd, dt, order_space, bnd);
    } else {
      LaunchKernel(L, &SemiLagrangeEulerOursMAC, flags,
                   flags, vel, src, fwd, dt, order_space, bnd, line_trace);
    }

    // Do the backwards step.
    if (method == ADVECT_MACCORMACK_MANTA) {
      LaunchKernel(L, &SemiLagrangeMAC, flags,
                   flags, vel, fwd, bwd, -dt, order_space, bnd);
    } else {
      LaunchKernel(L, &SemiLagrangeEulerOursMAC, flags,
                   flags, vel, fwd, bwd, -dt, order_space, bnd, line_trace);
    }

    // Perform the correction.
    LaunchKernel(L, &MacCormackCorrectMAC, flags,
                 flags, src, fwd, bwd, dst, maccormack_strength);
    
    // Perform clamping.
    // TODO(tompson): Perform our own clamping.
    LaunchKernel(L, &MacCormackClampMAC, flags,
                 flags, vel, dst, src, fwd, dt, bnd);
  } else {
    THError("Advection method not supported!");
  }

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
// addGravity
// *****************************************************************************

__global__ void addGravity(
    CudaFlagGrid flags, CudaMACGrid vel, THCDeviceTensor<float, 1> force,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    return;
  }

  const bool curFluid = flags.isFluid(i, j, k, b);
  const bool curEmpty = flags.isEmpty(i, j, k, b);

  if (!curFluid && !curEmpty) {
    return;
  }

  if (flags.isFluid(i - 1, j, k, b) ||
      (curFluid && flags.isEmpty(i - 1, j, k, b))) {
    vel(i, j, k, 0, b) += force[0];
  }

  if (flags.isFluid(i, j - 1, k, b) ||
      (curFluid && flags.isEmpty(i, j - 1, k, b))) {
    vel(i, j, k, 1, b) += force[1];
  }

  if (flags.is_3d() && (flags.isFluid(i, j, k - 1, b) ||
      (curFluid && flags.isEmpty(i, j, k - 1, b)))) {
    vel(i, j, k, 2, b) += force[2];
  }
}

static int tfluids_CudaMain_addGravity(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_gravity = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const float dt = static_cast<float>(lua_tonumber(L, 4));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 5));
  THCudaTensor* tensor_force = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));

  if (tensor_gravity->nDimension != 1 || tensor_gravity->size[0] != 3) {
    luaL_error(L, "ERROR: gravity must be a 3D vector (even in 2D)");
  }

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid vel = toCudaMACGrid(state, tensor_u, is_3d);

  const float mult = dt / flags.getDx();
  THCudaTensor_mul(state, tensor_force, tensor_gravity, mult);
  THCDeviceTensor<float, 1> force =
      toDeviceTensor<float, 1>(state, tensor_force);

  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &addGravity, flags,
               flags, vel, force, bnd);

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

