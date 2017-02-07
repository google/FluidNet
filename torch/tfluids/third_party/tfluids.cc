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

inline real SemiLagrange(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel, tfluids_(RealGrid)& src, 
    real dt, bool is_levelset, int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  const real p5 = static_cast<real>(0.5);
  tfluids_(vec3) pos =
      (tfluids_(vec3)((real)i + p5, (real)j + p5, (real)k + p5) -
       vel.getCentered(i, j, k, b) * dt);
  return src.getInterpolatedHi(pos, order_space, b);
}

inline real MacCormackCorrect(
    tfluids_(FlagGrid)& flags, const tfluids_(RealGrid)& old,
    const tfluids_(RealGrid)& fwd, const tfluids_(RealGrid)& bwd,
    const real strength, bool is_levelset, int32_t i, int32_t j, int32_t k,
    int32_t b) {
  real dst = fwd(i, j, k, b);

  if (flags.isFluid(i, j, k, b)) {
    // Only correct inside fluid region.
    dst += strength * 0.5 * (old(i, j, k, b) - bwd(i, j, k, b));
  }
  return dst;
}

inline void getMinMax(real& minv, real& maxv, const real& val) {
  if (val < minv) {
    minv = val;
  }
  if (val > maxv) {
    maxv = val;
  }
}

inline real clamp(const real val, const real min, const real max) {
  return std::min<real>(max, std::max<real>(min, val));
}

inline real doClampComponent(
    const Int3& gridSize, real dst, const tfluids_(RealGrid)& orig, real fwd,
    const tfluids_(vec3)& pos, const tfluids_(vec3)& vel, int32_t b) { 
  real minv = std::numeric_limits<real>::max();
  real maxv = -std::numeric_limits<real>::max();

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp(curr_pos.y, 0, gridSize.y - 1); 
    const int32_t k0 = clamp(curr_pos.z, 0, 
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

  dst = clamp(dst, minv, maxv);
  return dst;
}

inline real MacCormackClamp(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel, real dval,
    const tfluids_(RealGrid)& orig, const tfluids_(RealGrid)& fwd, real dt,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  Int3 gridUpper = flags.getSize() - 1;

  dval = doClampComponent(gridUpper, dval, orig, fwd(i, j, k, b),
                          tfluids_(vec3)(i, j, k),
                          vel.getCentered(i, j, k, b) * dt, b);

  // Lookup forward/backward, round to closest NB.
  Int3 pos_fwd = toInt3(tfluids_(vec3)(i, j, k) +
                        tfluids_(vec3)(0.5, 0.5, 0.5) -
                        vel.getCentered(i, j, k, b) * dt);
  Int3 pos_bwd = toInt3(tfluids_(vec3)(i, j, k) +
                        tfluids_(vec3)(0.5, 0.5, 0.5) +
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

  return dval;
}

static int tfluids_(Main_advectScalar)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* tensor_s =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_fwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  THTensor* tensor_bwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));
  const std::string method = static_cast<std::string>(lua_tostring(L, 8));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 9));
  THTensor* tensor_s_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 10, torch_Tensor));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) src(tensor_s, is_3d);
  tfluids_(RealGrid) dst(tensor_s_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  tfluids_(RealGrid) fwd(tensor_fwd, is_3d);
  tfluids_(RealGrid) bwd(tensor_bwd, is_3d); 


  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }
  const int32_t order = method == "euler" ? 1 : 2;
  const bool is_levelset = false;  // We never advect them.
  const int order_space = 1;

  const int32_t nbatch = flags.nbatch();
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();

  for (int32_t b = 0; b < nbatch; b++) {
    const int32_t bnd = 1;
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            if (order == 1) {
              dst(i, j, k, b) = 0;
            } else {
              fwd(i, j, k, b) = 0;
            }
            continue;
          }

          // Forward step.
          const real val = SemiLagrange(flags, vel, src, dt, is_levelset,
                                        order_space, i, j, k, b); 

          if (order == 1) {
            dst(i, j, k, b) = val;  // Store in the output array
          } else {
            fwd(i, j, k, b) = val;  // Store in the fwd array.
          }
        }
      }
    }

    if (order == 1) {
      // We're done. The forward Euler step is already in the output array.
    } else {

      // Otherwise we need to do the backwards step (which is a SemiLagrange
      // step on the forward data - hence we needed to finish the above loops
      // beforemoving on).
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd(i, j, k, b) = 0;
              continue; 
            } 

            // Backwards step.
            bwd(i, j, k, b) = SemiLagrange(flags, vel, fwd, -dt, is_levelset,
                                           order_space, i, j, k, b);
          }
        }
      }

      // Now compute the correction.
      const real strength = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) { 
            dst(i, j, k, b) = MacCormackCorrect(flags, src, fwd, bwd, strength,
                                                is_levelset, i, j, k, b);
          }
        }
      }

      // Now perform clamping.
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              continue;
            }
            const real dval = dst(i, j, k, b);
            dst(i, j, k, b) = MacCormackClamp(flags, vel, dval, src, fwd, dt,
                                              i, j, k, b);
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// advectVel
// *****************************************************************************

inline tfluids_(vec3) SemiLagrangeMAC(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel, tfluids_(MACGrid)& src,
    real dt, int order_space, int32_t i, int32_t j, int32_t k, int32_t b) {
  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  const tfluids_(vec3) pos(static_cast<real>(i) + 0.5,
                           static_cast<real>(j) + 0.5,
                           static_cast<real>(k) + 0.5);

  tfluids_(vec3) xpos = pos - vel.getAtMACX(i, j, k, b) * dt;
  const real vx = src.getInterpolatedComponentHi(xpos, order_space, 0, b);

  tfluids_(vec3) ypos = pos - vel.getAtMACY(i, j, k, b) * dt;
  const real vy = src.getInterpolatedComponentHi(ypos, order_space, 1, b);

  real vz;
  if (vel.is_3d()) {
    tfluids_(vec3) zpos = pos - vel.getAtMACZ(i, j, k, b) * dt;
    vz = src.getInterpolatedComponentHi(zpos, order_space, 2, b);
  } else {
    vz = 0;
  }

  return tfluids_(vec3)(vx, vy, vz);
}

inline tfluids_(vec3) MacCormackCorrectMAC(
    tfluids_(FlagGrid)& flags, const tfluids_(MACGrid)& old,
    const tfluids_(MACGrid)& fwd, const tfluids_(MACGrid)& bwd,
    const real strength, int32_t i, int32_t j, int32_t k, int32_t b) {
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

  tfluids_(vec3) dst(0, 0, 0);

  const int32_t dim = flags.is_3d() ? 3 : 2;
  for (int32_t c = 0; c < dim; ++c) {
    if (skip[c]) {
      dst(c) = fwd(i, j, k, c, b);
    } else {
      // perform actual correction with given strength.
      dst(c) = fwd(i, j, k, c, b) + strength * 0.5 * (old(i, j, k, c, b) -
                                                      bwd(i, j, k, c, b));
    }
  }

  return dst;
}

template <int32_t c>
inline real doClampComponentMAC(
    const Int3& gridSize, real dst, const tfluids_(MACGrid)& orig,
    real fwd, const tfluids_(vec3)& pos, const tfluids_(vec3)& vel,
    int32_t b) {
  real minv = std::numeric_limits<real>::max();
  real maxv = -std::numeric_limits<real>::max();

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp(curr_pos.y, 0, gridSize.y - 1);
    const int32_t k0 = clamp(curr_pos.z, 0,
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

  dst = clamp(dst, minv, maxv);
  return dst;
}

inline tfluids_(vec3) MacCormackClampMAC(
    tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel,
    tfluids_(vec3) dval, const tfluids_(MACGrid)& orig,
    const tfluids_(MACGrid)& fwd, real dt,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  tfluids_(vec3) pos(static_cast<real>(i), static_cast<real>(j),
                     static_cast<real>(k));
  tfluids_(vec3) dfwd = fwd(i, j, k, b);
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
  
  return dval;
}

static int tfluids_(Main_advectVel)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  const real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_fwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_bwd =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 6));
  const std::string method = static_cast<std::string>(lua_tostring(L, 7));
  const int32_t boundary_width = static_cast<int32_t>(lua_tointeger(L, 8));
  THTensor* tensor_u_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 9, torch_Tensor));

  if (method != "maccormack" && method != "euler") {
    luaL_error(L, "advectScalar method is not supported.");
  }
  const int32_t order = method == "euler" ? 1 : 2;
  const int order_space = 1;

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);

  // We always do self-advection, but we could point orig to another tensor.
  tfluids_(MACGrid) orig(tensor_u, is_3d);
  tfluids_(MACGrid) dst(tensor_u_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  tfluids_(MACGrid) fwd(tensor_fwd, is_3d);
  tfluids_(MACGrid) bwd(tensor_bwd, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            if (order == 1) {
              dst.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            } else {
              fwd.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            }
            continue;
          }

          // Forward step.
          tfluids_(vec3) val = SemiLagrangeMAC(flags, vel, orig, dt,
                                               order_space, i, j, k, b); 

          if (order == 1) {
            dst.setSafe(i, j, k, b, val);  // Store in the output array
          } else {
            fwd.setSafe(i, j, k, b, val);  // Store in the fwd array.
          }
        }
      }
    }

    if (order == 1) {
      // We're done. The forward Euler step is already in the output array.
    } else {

      // Otherwise we need to do the backwards step (which is a SemiLagrange
      // step on the forward data - hence we needed to finish the above loops
      // before moving on).
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
              continue; 
            } 

            // Backwards step.
            bwd.setSafe(i, j, k, b, SemiLagrangeMAC(
                flags, vel, fwd, -dt, order_space, i, j, k, b));
          }
        }
      }

      // Now compute the correction.
      const real strength = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) { 
            dst.setSafe(i, j, k, b, MacCormackCorrectMAC(
                flags, orig, fwd, bwd, strength, i, j, k, b));
          }
        }
      }

      // Now perform clamping.
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              continue;
            }
            const tfluids_(vec3) dval = dst(i, j, k, b);
            dst.setSafe(i, j, k, b, MacCormackClampMAC(
                flags, vel, dval, orig, fwd, dt, i, j, k, b));
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

static int tfluids_(Main_setWallBcsForward)(lua_State *L) {
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          const bool cur_fluid = flags.isFluid(i, j, k, b);
          const bool cur_obs = flags.isObstacle(i, j, k, b);
          if (!cur_fluid && !cur_obs) {
            continue;
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
      }
    } 
  }

  return 0;  // Recall: number of return values on the lua stack.
}

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

static int tfluids_(Main_velocityDivergenceForward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_u_div =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) rhs(tensor_u_div, is_3d);


  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    // Note: our kernel assumes enforceCompatibility == false (i.e. we do not
    // do the reduction) and that fractions are not provided.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            rhs(i, j, k, b) = 0;
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            rhs(i, j, k, b) = 0;
            continue;
          }

          // compute divergence 
          // no flag checks: assumes vel at obstacle interfaces is set to zero.
          real div = 
              vel(i, j, k, 0, b) - vel(i + 1, j, k, 0, b) +
              vel(i, j, k, 1, b) - vel(i, j + 1, k, 1, b);
          if (is_3d) {
            div += (vel(i, j, k, 2, b) - vel(i, j, k + 1, 2, b));
          }
          rhs(i, j, k, b) = div;
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// velocityUpdateForward
// *****************************************************************************

static int tfluids_(Main_velocityUpdateForward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));
 
  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) pressure(tensor_p, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize(); 
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd || 
              j < bnd || j > ysize - 1 - bnd || 
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta doesn't touch the velocity on the boundaries (i.e.
            // it stays constant).
            continue;
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
            if (is_3d && flags.isFluid(i, j, k - 1, b)) {
              vel(i, j, k, 2, b) -= (pressure(i, j, k, b) -
                                     pressure(i, j, k - 1, b));
            }
      
            if (flags.isEmpty(i - 1, j, k, b)) {
              vel(i, j, k, 0, b) -= pressure(i, j, k, b);
            }
            if (flags.isEmpty(i, j - 1, k, b)) {
              vel(i, j, k, 1, b) -= pressure(i, j, k, b);
            }
            if (is_3d && flags.isEmpty(i, j, k - 1, b)) {
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
            if (is_3d) {
              if (flags.isFluid(i, j, k - 1, b)) {
                vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
              } else {
                vel(i, j, k, 2, b)  = 0.f;
              }
            }
          }
        }
      }
    }
  }
  
  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// addBuoyancy
// *****************************************************************************
  
static int tfluids_(Main_addBuoyancy)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_density =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_gravity =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_strength =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  const real dt = static_cast<real>(lua_tonumber(L, 6));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));

  if (tensor_gravity->nDimension != 1 || tensor_gravity->size[0] != 3) {
    luaL_error(L, "ERROR: gravity must be a 3D vector (even in 2D)");
  }
  const real* gdata = THTensor_(data)(tensor_gravity);

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(RealGrid) factor(tensor_density, is_3d);

  // Note: We wont use the tensor_strength temp space for the C++ version.
  // It's just as fast (and easy) for us to wrap in a vec3.
  static_cast<void>(tensor_strength);
  const tfluids_(vec3) strength =
      tfluids_(vec3)(-gdata[0], -gdata[1], -gdata[2]) * (dt / flags.getDx());

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    // Note: our kernel assumes enforceCompatibility == false (i.e. we do not
    // do the reductiion) and that fractions are not provided.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // No buoyancy on the border.
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            continue;
          }
          if (flags.isFluid(i - 1, j, k, b)) {
            vel(i, j, k, 0, b) += (static_cast<real>(0.5) * strength.x *
                                (factor(i, j, k, b) + factor(i - 1, j, k, b)));
          }
          if (flags.isFluid(i, j - 1, k, b)) {
            vel(i, j, k, 1, b) += (static_cast<real>(0.5) * strength.y *
                                (factor(i, j, k, b) + factor(i, j - 1, k, b)));
          }
          if (is_3d && flags.isFluid(i, j, k - 1, b)) {
            vel(i, j, k, 2, b) += (static_cast<real>(0.5) * strength.z *
                                (factor(i, j, k, b) + factor(i, j, k - 1, b)));
          }
        }
      }
    }
  }
  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// vorticityConfinement
// *****************************************************************************

inline void AddForceField(
    const tfluids_(FlagGrid)& flags, tfluids_(MACGrid)& vel,
    const tfluids_(VecGrid)& force, int32_t i, int32_t j, int32_t k,
    int32_t b) {
  const bool curFluid = flags.isFluid(i, j, k, b);
  const bool curEmpty = flags.isEmpty(i, j, k, b);
  if (!curFluid && !curEmpty) {
    return;
  }

  if (flags.isFluid(i - 1, j, k, b) || 
      (curFluid && flags.isEmpty(i - 1, j, k, b))) {
    vel(i, j, k, 0, b) += (static_cast<real>(0.5) *
                        (force(i - 1, j, k, 0, b) + force(i, j, k, 0, b)));
  }

  if (flags.isFluid(i, j - 1, k, b) ||
      (curFluid && flags.isEmpty(i, j - 1, k, b))) {
    vel(i, j, k, 1, b) += (static_cast<real>(0.5) * 
                        (force(i, j - 1, k, 1, b) + force(i, j, k, 1, b)));
  }

  if (flags.is_3d() && (flags.isFluid(i, j, k - 1, b) ||
      (curFluid && flags.isEmpty(i, j, k - 1, b)))) {
    vel(i, j, k, 2, b) += (static_cast<real>(0.5) *
                        (force(i, j, k - 1, 2, b) + force(i, j, k, 2, b)));
  }
}

static int tfluids_(Main_vorticityConfinement)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  const real strength = static_cast<real>(lua_tonumber(L, 3));
  THTensor* tensor_centered =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tensor_curl =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  THTensor* tensor_curl_norm =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));
  THTensor* tensor_force =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 7, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 8));


  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(MACGrid) vel(tensor_u, is_3d);
  tfluids_(VecGrid) centered(tensor_centered, is_3d);
  tfluids_(VecGrid) curl(tensor_curl, true);  // Alawys 3D.
  tfluids_(RealGrid) curl_norm(tensor_curl_norm, is_3d);
  tfluids_(VecGrid) force(tensor_force, is_3d);
  
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    // First calculate the centered velocity.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            centered.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            continue;
          }
          centered.setSafe(i, j, k, b, vel.getCentered(i, j, k, b));
        }
      }
    }

    // Now calculate the curl and it's (l2) norm.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            curl.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            curl_norm(i, j, k, b) = 0;
            continue;
          }

          // Calculate the curl and it's (l2) norm.
          const tfluids_(vec3) cur_curl(centered.curl(i, j, k, b));
          curl.setSafe(i, j, k, b, cur_curl);
          curl_norm(i, j, k, b) = cur_curl.norm();
        }
      } 
    }

    // Now calculate the vorticity confinement force.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd || 
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Don't add force on the boundaries.
            force.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
            continue;
          }

          tfluids_(vec3) grad(0, 0, 0);
          grad.x = static_cast<real>(0.5) * (curl_norm(i + 1, j, k, b) -
                                             curl_norm(i - 1, j, k, b));
          grad.y = static_cast<real>(0.5) * (curl_norm(i, j + 1, k, b) -
                                             curl_norm(i, j - 1, k, b));
          if (is_3d) {
            grad.z = static_cast<real>(0.5) * (curl_norm(i, j, k + 1, b) -
                                               curl_norm(i, j, k - 1, b));
          }
          grad.normalize();
          
          force.setSafe(i, j, k, b, tfluids_(vec3)::cross(
              grad, curl(i, j, k, b)) * strength);
        }   
      }
    }

    // Now apply the force.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            continue;
          }
          AddForceField(flags, vel, force, i, j, k, b);          
        }  
      }
    } 
  }

  return 0;  // Recall: number of return values on the lua stack. 
}
