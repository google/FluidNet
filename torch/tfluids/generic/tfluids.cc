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

#ifndef TH_GENERIC_FILE
  #define TH_GENERIC_FILE "generic/tfluids.cc"
#else

#include <assert.h>
#include <memory>

#include "generic/vec3.cc"
#include "generic/calc_line_trace.cc"

#ifdef BUILD_GL_FUNCS
  #if defined (__APPLE__) || defined (OSX)
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <OpenGL/glext.h>
  #else
    #include <GL/gl.h>
  #endif

  #ifndef GLUT_API_VERSION
    #if defined(macintosh) || defined(__APPLE__) || defined(OSX)
      #include <GLUT/glut.h>
    #elif defined (__linux__) || defined (UNIX) || defined(WIN32) || defined(_WIN32)
      #include "GL/glut.h"
    #endif
  #endif
#endif

static inline real Mix(const real a, const real b, const real t) {
  return (static_cast<real>(1.0) - t) * a + t * b;
}

static inline void MixWithGeom(
    const real a, const real b, const bool a_geom, const bool b_geom,
    const bool sample_into_geom, const real t, real* interp_val,
    bool* interp_geom) {
  if (sample_into_geom || (!a_geom && !b_geom)) {
    *interp_geom = false;
    *interp_val = (static_cast<real>(1.0) - t) * a + t * b;
  } else if (a_geom && !b_geom) {
    *interp_geom = false;
    *interp_val = b;  // a is geometry, return b.
  } else if (b_geom && !a_geom) {
    *interp_geom = false;
    *interp_val = a;  // b is geometry, return a.
  } else {
    *interp_geom = true;  // both a and b are geom.
    *interp_val = static_cast<real>(0);
  }
}

static inline real ClampReal(const real x, const real low, const real high) {
  return std::max<real>(std::min<real>(x, high), low);
}

static inline real GetData(const real* x, int32_t i, int32_t j, int32_t k,
                           const Int3& dim) {
  if (tfluids_(IsOutOfDomain)(i, j, k, dim)) {
    printf("ERROR: GetData index is out of domain\n");
    exit(-1);
  }
  return x[IX(i, j, k, dim)];
}

// NOTE: As per our CalcLineTrace method we define the integer position values
// as THE CENTER of the grid cells. That means that a value of (0, 0, 0) is the
// center of the first grid cell, so the (-0.5, 0, 0) position is the LEFT
// face of the first cell. Likewise (xdim - 0.5, ydim - 0.5, zdim - 0.5) is the
// upper bound of the grid (right at the corner).
//
// You should NEVER call this function on a grid cell that is either
// GEOMETRY (obs[pos] == 1) or touching the grid domain. If this is the case
// then likely the CalcLineTrace function failed and this is a logic bug
// (CalcLineTrace should stop the line trace BEFORE going into an invalid
// region).
//
// If sample_into_geom is true then we will do bilinear interpolation into
// neighboring geometry cells (this is done during velocity advection since we
// set the internal geom velocities to zero out geometry face velocity).
// Otherwise, we will clamp the scalar field value at the non-geometry boundary.
static inline real tfluids_(Main_GetInterpValue)(
    const real * x, const real* obs, const tfluids_(vec3)& pos,
    const Int3& dims, const bool sample_into_geom) {

  // TODO(tompson,kris): THIS ASSUMES THAT OBSTACLES HAVE ZERO VELOCITY.

  // Make sure we're not against the grid boundary or beyond it.
  // This is a conservative test (i.e. we test if position is on or beyond it).
  if (IsOutOfDomainReal(pos, dims)) {
    printf("ERROR: GetInterpValue called on a point (%f, %f, %f) out of the "
           "domain!\n", pos.x, pos.y, pos.z);
    exit(-1);
  }

  // Get the current integer location of the pixel.
  int32_t i0, j0, k0;
  GetPixelCenter(pos, &i0, &j0, &k0);
  if (tfluids_(IsOutOfDomain(i0, j0, k0, dims))) {
    printf("ERROR: GetInterpValue pixel center is out of domain!\n");
    exit(-1);
  }

  // The current center SHOULD NOT be geometry.
  if (IsBlockedCell(obs, i0, j0, k0, dims)) {
    printf("ERROR: GetInterpValue called on a blocked cell!\n");
    exit(-1);
  }

  // Calculate the next cell integer to interpolate with AND calculate the
  // interpolation coefficient.

  // If we're on the left hand size of the grid center we should be
  // interpolating left (p0) to center (p1), and if we're on the right we
  // should be interpolating center (p0) to right (p1).
  // RECALL: (0,0) is defined as the CENTER of the first cell. (xdim - 1,
  // ydim - 1) is defined as the center of the last cell.
  real icoef, jcoef, kcoef;
  int32_t i1, j1, k1;
  if (pos.x < static_cast<real>(i0)) {
    i1 = i0;
    i0 = std::max<int32_t>(i0 - 1, 0);
  } else {
    i1 = std::min<int32_t>(i0 + 1, dims.x - 1);
  }
  icoef = (i0 == i1) ? static_cast<real>(0) : pos.x - static_cast<real>(i0);

  // Same logic for top / bottom and front / back interp.
  if (pos.y < static_cast<real>(j0)) {
    j1 = j0;
    j0 = std::max<int32_t>(j0 - 1, 0);
  } else {
    j1 = std::min<int32_t>(j0 + 1, dims.y - 1);
  }
  jcoef = (j0 == j1) ? static_cast<real>(0) : pos.y - static_cast<real>(j0);

  if (pos.z < static_cast<real>(k0)) {
    k1 = k0;
    k0 = std::max<int32_t>(k0 - 1, 0);
  } else {
    k1 = std::min<int32_t>(k0 + 1, dims.z - 1);
  }
  kcoef = (k0 == k1) ? static_cast<real>(0) : pos.z - static_cast<real>(k0);

  if (!(icoef >= 0 && icoef <= 1 && jcoef >= 0 && jcoef <= 1 &&
        kcoef >= 0 && kcoef <= 1)) {
    printf("ERROR: mixing coefficients are not in [0, 1]\n");
    printf("  icoef, jcoef, kcoef = %f, %f, %f.\n", icoef, jcoef, kcoef);
    exit(-1);
  }

  if (tfluids_(IsOutOfDomain)(i0, j0, k0, dims) ||
      tfluids_(IsOutOfDomain)(i1, j1, k1, dims)) {
    printf("ERROR: interpolation coordinates are out of the domain!.\n");
    printf("  i0, j0, k0, i1, j1, k1 = %d, %d, %d, %d, %d, %d.\n",
           i0, j0, k0, i1, j1, k1);
    exit(-1);
  }

  // Note: we DO NOT need to handle geometry when doing the trilinear
  // interpolation:
  // 1. The particle trace is gaurenteed to return a position that is NOT within
  // geometry (but can be epsilon away from a geometry wall).
  // 2. We have called setObstacleBcs prior to running advection which
  // allows us to interpolate one level into geometry without violating
  // constraints (it does so by setting geometry velocities such that the face
  // velocity is zero).

  // Assume a cube with 8 points.
  // Front face.
  // Top Front MIX.
  const real xFrontLeftTop = x[IX(i0, j1, k0, dims)];
  const bool gFrontLeftTop = IsBlockedCell(obs, i0, j1, k0, dims);
  const real xFrontRightTop =  x[IX(i1, j1, k0, dims)];
  const bool gFrontRightTop = IsBlockedCell(obs, i1, j1, k0, dims);
  real xFrontTopInterp;
  bool gFrontTopInterp;
  MixWithGeom(xFrontLeftTop, xFrontRightTop, gFrontLeftTop, gFrontRightTop,
      sample_into_geom, icoef, &xFrontTopInterp, &gFrontTopInterp);

  // Bottom Front MIX.
  const real xFrontLeftBottom = x[IX(i0, j0, k0, dims)];
  const bool gFrontLeftBottom = IsBlockedCell(obs, i0, j0, k0, dims);
  const real xFrontRightBottom = x[IX(i1, j0, k0, dims)];
  const bool gFrontRightBottom = IsBlockedCell(obs, i1, j0, k0, dims);
  real xFrontBottomInterp;
  bool gFrontBottomInterp;
  MixWithGeom(
    xFrontLeftBottom, xFrontRightBottom, gFrontLeftBottom, gFrontRightBottom,
    sample_into_geom, icoef, &xFrontBottomInterp, &gFrontBottomInterp);

  // Back face.
  // Top Back MIX.
  const real xBackLeftTop = x[IX(i0, j1, k1, dims)];
  const bool gBackLeftTop = IsBlockedCell(obs, i0, j1, k1, dims);
  const real xBackRightTop = x[IX(i1, j1, k1, dims)];
  const bool gBackRightTop = IsBlockedCell(obs, i1, j1, k1, dims);
  real xBackTopInterp;
  bool gBackTopInterp;
  MixWithGeom(xBackLeftTop, xBackRightTop, gBackLeftTop, gBackRightTop,
      sample_into_geom, icoef, &xBackTopInterp, &gBackTopInterp);

  // Bottom Back MIX.
  const real xBackLeftBottom = x[IX(i0, j0, k1, dims)];
  const bool gBackLeftBottom = IsBlockedCell(obs, i0, j0, k1, dims);
  const real xBackRightBottom = x[IX(i1, j0, k1, dims)];
  const bool gBackRightBottom = IsBlockedCell(obs, i1, j0, k1, dims);
  real xBackBottomInterp;
  bool gBackBottomInterp;
  MixWithGeom(
      xBackLeftBottom, xBackRightBottom, gBackLeftBottom, gBackRightBottom,
      sample_into_geom, icoef, &xBackBottomInterp, &gBackBottomInterp);

  // Now get middle of front - The bilinear interp of the front face.
  real xBiLerpFront;
  bool gBiLerpFront;
  MixWithGeom(
      xFrontBottomInterp, xFrontTopInterp, gFrontBottomInterp, gFrontTopInterp,
      sample_into_geom, jcoef, &xBiLerpFront, &gBiLerpFront);

  // Now get middle of back - The bilinear interp of the back face.
  real xBiLerpBack;
  bool gBiLerpBack;
  MixWithGeom(
      xBackBottomInterp, xBackTopInterp, gBackBottomInterp, gBackTopInterp,
      sample_into_geom, jcoef, &xBiLerpBack, &gBiLerpBack);

  // Now get the interpolated point between the points calculated in the front
  // and back faces - The trilinear interp part.
  real xTriLerp;
  bool gTriLerp;
  MixWithGeom(xBiLerpFront, xBiLerpBack, gBiLerpFront, gBiLerpBack,
              sample_into_geom, kcoef, &xTriLerp, &gTriLerp);

  // At least ONE of the samples shouldn't have been geometry so the final value
  // should be valid.
  if (gTriLerp) {
    printf("ERROR: Couldn't find a non-blocked cell. Logic is broken.!\n");
    exit(-1);
  }

  return xTriLerp;
}

static inline void getCurl3D(
    const real * u, const real * v, const real * w, const int i, const int j,
    const int k, const Int3& dims, tfluids_(vec3)* curl) {
  if (tfluids_(IsOutOfDomain(i, j, k, dims))) {
    printf("Error: getCurl3D called on out of domain index.\n");
    exit(-1);
  }
  if (w == nullptr) {
    printf("Error: getCurl3D w component is null.\n");
    exit(-1);
  }
  const real half = static_cast<real>(0.5);

  real dwdj, dudj;
  if (j == 0) {
    // Single sided diff (pos side).
    dwdj = GetData(w, i, j + 1, k, dims) - GetData(w, i, j, k, dims);
    dudj = GetData(u, i, j + 1, k, dims) - GetData(u, i, j, k, dims);
  } else if (j == dims.y - 1) {
    // Single sided diff (neg side).
    dwdj = GetData(w, i, j, k, dims) - GetData(w, i, j - 1, k, dims); 
    dudj = GetData(u, i, j, k, dims) - GetData(u, i, j - 1, k, dims);
  } else {
    // Central diff.
    dwdj = half * (GetData(w, i, j + 1, k, dims) -
        GetData(w, i, j - 1, k, dims));
    dudj = half * (GetData(u, i, j + 1, k, dims) -
        GetData(u, i, j - 1, k, dims));
  }

  real dwdi, dvdi;
  if (i == 0) {
    // Single sided diff (pos side).
    dwdi = GetData(w, i + 1, j, k, dims) - GetData(w, i, j, k, dims);
    dvdi = GetData(v, i + 1, j, k, dims) - GetData(v, i, j, k, dims);
  } else if (i == dims.x - 1) {
    // Single sided diff (neg side).
    dwdi = GetData(w, i, j, k, dims) - GetData(w, i - 1, j, k, dims);
    dvdi = GetData(v, i, j, k, dims) - GetData(v, i - 1, j, k, dims);
  } else {
    // Central diff.
    dwdi = half * (GetData(w, i + 1, j, k, dims) -
        GetData(w, i - 1, j, k, dims));
    dvdi = half * (GetData(v, i + 1, j, k, dims) -
        GetData(v, i - 1, j, k, dims));
  }

  real dudk, dvdk;
  if (k == 0) {
    // Single sided diff (pos side).
    dudk = GetData(u, i, j, k + 1, dims) - GetData(u, i, j, k, dims);
    dvdk = GetData(v, i, j, k + 1, dims) - GetData(v, i, j, k, dims);
  } else if (k == dims.z - 1) {
    // Single sided diff (neg side).
    dudk = GetData(u, i, j, k, dims) - GetData(u, i, j, k - 1, dims);
    dvdk = GetData(v, i, j, k, dims) - GetData(v, i, j, k - 1, dims);
  } else {
    // Central diff.
    dudk = half * (GetData(u, i, j, k + 1, dims) -
        GetData(u, i, j, k - 1, dims));
    dvdk = half * (GetData(v, i, j, k + 1, dims) -
        GetData(v, i, j, k - 1, dims));
  }

  curl->x = dwdj - dvdk;
  curl->y = dudk - dwdi;
  curl->z = dvdi - dudj;
}

static inline real getCurl2D(
    const real * u, const real * v, const int i, const int j,
    const Int3& dims) {
  if (dims.z != 1) {
    printf("Error: input dimensions are not 2D.\n");
    exit(-1);
  }
  if (tfluids_(IsOutOfDomain(i, j, 0, dims))) {
    printf("Error: getCurl2D called on out of bounds index.\n");
    exit(-1);
  }
  const real half = static_cast<real>(0.5);
  const int32_t k = 0;
  
  real dvdi;
  if (i == 0) {
    // Single sided diff (pos side).
    dvdi = GetData(v, i + 1, j, k, dims) - GetData(v, i, j, k, dims);
  } else if (i == dims.x - 1) {
    // Single sided diff (neg side).
    dvdi = GetData(v, i, j, k, dims) - GetData(v, i - 1, j, k, dims);
  } else {
    // Central diff.
    dvdi = half * (GetData(v, i + 1, j, k, dims) -
        GetData(v, i - 1, j, k, dims));
  }

  real dudj;
  if (j == 0) {
    // Single sided diff (pos side).
    dudj = GetData(u, i, j + 1, k, dims) - GetData(u, i, j, k, dims);
  } else if (j == dims.y - 1) {
    // Single sided diff (neg side).
    dudj = GetData(u, i, j, k, dims) - GetData(u, i, j - 1, k, dims);
  } else {
    // Central diff.
    dudj = half * (GetData(u, i, j + 1, k, dims) -
        GetData(u, i, j - 1, k, dims));
  }

  return dudj - dvdi;
}

// This function is taken from CNNFluids/fluid_solver/second_order_solver.h and
// modified.
//
// @param dt - Time step.
// @param scale - scale of confinement force.  Higher is stronger/more curl.
// @param u, v, w - The input velocity field, from which force 
// and curl will be calculated. Note: param w is OPTIONAL, it can be nullptr 
// when we're using 2D systems.
// @param obs - The input occupancy grid (1.0f == occupied, 0.0f == empty).
// @param dims - The input dimension {x=xdim, y=ydim, z=zdim}.
// @param curl_u, curl_v, curl_w, mag_curl - Temporary space the same size as
// u, v, w.
//
// Note: param w is OPTIONAL, it can be nullptr when we're using 2D systems.
//
// Note: All fields are assumed to be laid out xdim first, ydim second, zdim
// third (outer), see IX() for index lookup.
static void tfluids_(Main_vorticityConfinement)(
  const real dt, const real scale, real* u, real* v, real* w, const real* obs,
  const Int3& dims, real* curl_u, real* curl_v, real* curl_w, real* mag_curl) {
  const bool two_dim = w == nullptr;
  if (!(two_dim == (curl_w == nullptr))) {
    printf("Error: vorticityConfinement two_ddim does not match 'w' input.\n");
    exit(-1);
  }

  const real half = static_cast<real>(0.5);

  if (!(dims.x > 2 && dims.y > 2)) {
    printf("Error: vorticityConfinement x and y dim are too small.\n");
    exit(-1);
  }
  if (!two_dim && dims.z <= 2) {
    printf("Error: vorticityConfinement z dim is too small.\n");
    exit(-1);
  }

  // As a pre-step, calculate curl in each cell.
  int32_t k, j, i;
  #pragma omp parallel for collapse(3) private(k, j, i)
  for (k = 0; k < dims.z; k++) {
    for (j = 0; j < dims.y; j++) {
      for (i = 0; i < dims.x; i++) {
        if (two_dim) {
          // The curl in 2D is a scalar value.
          const real curl = getCurl2D(u, v, i, j, dims);
          curl_u[IX(i, j, k, dims)] = curl;
          // L2 magnitude of a scalar field is just L1.
          mag_curl[IX(i, j, k, dims)] = std::fabs(curl);
        } else {
          // The curl in 3D is a vector value.
          tfluids_(vec3) curl;
          getCurl3D(u, v, w, i, j, k, dims, &curl);
          curl_u[IX(i, j, k, dims)] = curl.x;
          curl_v[IX(i, j, k, dims)] = curl.y;
          curl_w[IX(i, j, k, dims)] = curl.z;
          mag_curl[IX(i, j, k, dims)] = tfluids_(length3)(curl);
        }
      }
    }
  }

  // Now perform the vorticity confinement for internal cells (ignore
  // border cells).
  // TODO(tompson,kris): Handle geometry properly.
  // TODO(tompson,kris): You can also do the internal cells because we have
  // single sided FEM approx.
  
  const int kStart = two_dim ? 0 : 1;
  const int kEnd = two_dim ? 1 : (dims.z - 1);

#pragma omp parallel for collapse(3) private(k, j, i)
  for (k = kStart; k < kEnd; k++) {
    for (j = 1; j < dims.y - 1; j++) {
      for (i = 1; i < dims.x - 1; i++) {
        // Don't perform any confinement in an obstacle.
        if (IsBlockedCell(obs, i, j, k, dims)) {
          continue;
        }
        // Don't perform any confinement in a cell that borders an obstacle.
        // TODO(tompson): Not sure if this is correct. It's probably OK for now
        // only because the partial derivative for ||w|| is invalid for cells
        // that lie next to a geometry cell.
        if (IsBlockedCell(obs, i - 1, j, k, dims) ||
            IsBlockedCell(obs, i + 1, j, k, dims) ||
            IsBlockedCell(obs, i, j - 1, k, dims) ||
            IsBlockedCell(obs, i, j + 1, k, dims) ||
            (!two_dim && IsBlockedCell(obs, i, j, k - 1, dims)) ||
            (!two_dim && IsBlockedCell(obs, i, j, k + 1, dims))) {
          continue;
        }

        tfluids_(vec3) force;
        if (two_dim) {
          // Find derivative of the magnitude of curl (n = grad |w|).
          // Where 'w' is the curl calculated above.
          real dwdi = half * (mag_curl[IX(i + 1, j, k, dims)] -
              mag_curl[IX(i - 1, j, k, dims)]);
          real dwdj = half * (mag_curl[IX(i, j + 1, k, dims)] -
              mag_curl[IX(i, j - 1, k, dims)]);

          const real length_sq = dwdi * dwdi + dwdj * dwdj;
          const real length =
              (length_sq > std::numeric_limits<real>::epsilon()) ?
              std::sqrt(length_sq) : static_cast<real>(0);
          if (length > static_cast<real>(1e-6)) {
            dwdi /= length;
            dwdj /= length;
          }

          const real v = curl_u[IX(i, j, k, dims)];

          // N x w.
          force.x = dwdj * (-v);
          force.y = dwdi * v;
          force.z = static_cast<real>(0);
        } else {
          // Find derivative of the magnitude of curl (n = grad |w|).
          // Where 'w' is the curl calculated above.
         
          // Calculate magnitude gradient (grad |w|) using FEM central diff.
          real dwdi = half * (mag_curl[IX(i + 1, j, k, dims)] -
              mag_curl[IX(i - 1, j, k, dims)]);
          real dwdj = half * (mag_curl[IX(i, j + 1, k, dims)] -
              mag_curl[IX(i, j - 1, k, dims)]);
          real dwdk = half * (mag_curl[IX(i, j, k + 1, dims)] -
              mag_curl[IX(i, j, k - 1, dims)]);

          const real length_sq = dwdi * dwdi + dwdj * dwdj + dwdk * dwdk;
          const real length =
              (length_sq > std::numeric_limits<real>::epsilon()) ?
              std::sqrt(length_sq) : static_cast<real>(0);
          if (length > static_cast<real>(1e-6)) {
            dwdi /= length;
            dwdj /= length;
            dwdk /= length;
          }

          tfluids_(vec3) N;
          N.x = dwdi;
          N.y = dwdj;
          N.z = dwdk;

          tfluids_(vec3) curl;
          curl.x = curl_u[IX(i, j, k, dims)];
          curl.y = curl_v[IX(i, j, k, dims)];
          curl.z = curl_w[IX(i, j, k, dims)];

          tfluids_(cross3)(N, curl, &force);
        }

        // Now apply the force.
        u[IX(i, j, k, dims)] += force.x * scale * dt;
        v[IX(i, j, k, dims)] += force.y * scale * dt;
        if (!two_dim) {
          w[IX(i, j, k, dims)] += force.z * scale * dt;
        }
      }
    }
  }
}

// A hard-coded Gaussian kernel. Use AveKernel.m to calculate the values.
#ifndef TFLUIDS_KERNEL
#define TFLUIDS_KERNEL
const int32_t kern_rad = 1;
const int32_t kern_sz = kern_rad * 2 + 1;
const int32_t kern_numel = kern_sz * kern_sz * kern_sz;
#endif
const real tfluids_(kernel)[kern_numel] = {
    6.5880001e-05, 1.4994219e-03, 6.5880001e-05,
    1.4994219e-03, 3.4126686e-02, 1.4994219e-03,
    6.5880001e-05, 1.4994219e-03, 6.5880001e-05,
    1.4994219e-03, 3.4126686e-02, 1.4994219e-03,
    3.4126686e-02, 7.7671978e-01, 3.4126686e-02,
    1.4994219e-03, 3.4126686e-02, 1.4994219e-03,
    6.5880001e-05, 1.4994219e-03, 6.5880001e-05,
    1.4994219e-03, 3.4126686e-02, 1.4994219e-03,
    6.5880001e-05, 1.4994219e-03, 6.5880001e-05};

// The final sampling of the velocity field (after doing the backwards trace).
static real sampleField(const real* field, const tfluids_(vec3)& pos,
                        const real* obs, const Int3& dims,
                        const bool sample_into_geom) {
  return tfluids_(Main_GetInterpValue)(field, obs, pos, dims, sample_into_geom);
}

// Advect a scalar field along the velocity field using a first order
// Semi-Lagrangian (Euler) step.
//
// @param dt - Time step.
// @param q_src - The input scalar field to advect.
// @param u, v, w - The input velocity field, from which to advect q_src. Note:
// param w is OPTIONAL, it can be nullptr when we're advecting 2D systems.
// @param obs - The input occupancy grid (1.0f == occupied, 0.0f == empty).
// @param dims - The input dimension {x=xdim, y=ydim, z=zdim}.
// @param q_dst - The output advected scalar field.
//
// Note: All fields are assumed to be laid out xdim first, ydim second, zdim
// third (outer), see IX() for index lookup.
static void tfluids_(Main_advectScalarEuler)(
    const real dt, const real* q_src, const real* u, const real* v,
    const real* w, const real* obs, const Int3& dims, real* q_dst,
    const bool sample_into_geom) {
  int32_t k, j, i;
  const real ndt = -dt;
  const bool two_dim = w == nullptr;
#pragma omp parallel for collapse(3) private(k, j, i)
  for(k = 0; k < dims.z; k++) {
    for(j = 0; j < dims.y; j++) {
      for(i = 0; i < dims.x; i++) {
        if (IsBlockedCell(obs, i, j, k, dims)) {
          // Don't advect blocked cells.
          continue;
        }

        // NOTE: The integer positions are in the center of the grid cells.
        tfluids_(vec3) pos;
        pos.x = static_cast<real>(i);
        pos.y = static_cast<real>(j);
        pos.z = static_cast<real>(k);

        // Velocity is in grids / second.
        tfluids_(vec3) vel;
        vel.x = u[IX(i, j, k, dims)];
        vel.y = v[IX(i, j, k, dims)];
        vel.z = two_dim ? static_cast<real>(0) : w[IX(i, j, k, dims)];

        // Backtrace based upon current velocity at cell center.
        tfluids_(vec3) back_pos;
        tfluids_(vec3) displacement;
        displacement.x = (ndt * vel.x);
        displacement.y = (ndt * vel.y);
        displacement.z = (ndt * vel.z);
        // Step along the displacement vector. calcLineTrace will handle
        // boundary conditions for us. Note: it will terminate BEFORE the
        // boundary (i.e. the returned position is always valid).
        const bool hit_boundary = calcLineTrace(pos, displacement, dims, obs,
                                                back_pos);

        // Check the return value from calcLineTrace just in case.
        if (IsOutOfDomainReal(back_pos, dims) ||
            IsBlockedCellReal(obs, back_pos, dims)) {
          printf("Error: advectScalarEuler - calcLineTrace returned an "
                 "invalid cell (back_pos).\n");
          exit(-1);
        }


        // Finally, sample the value at the new position.
        q_dst[IX(i, j, k, dims)] = sampleField(q_src, back_pos, obs, dims,
                                               sample_into_geom);
      }
    }
  }
}

// Second order version of the above function.
static void tfluids_(Main_advectScalarRK2)(
    const real dt, const real* q_src, const real* u, const real* v,
    const real* w, const real* obs, const Int3& dims, real* q_dst,
    const bool sample_into_geom) {
  int32_t k, j, i;
  const real ndt = -dt;
  const bool two_dim = w == nullptr;
#pragma omp parallel for collapse(3) private(k, j, i)
  for(k = 0; k < dims.z; k++) { 
    for(j = 0; j < dims.y; j++) { 
      for(i = 0; i < dims.x; i++) {
        if (IsBlockedCell(obs, i, j, k, dims)) {
          // Don't advect blocked cells.
          continue;
        }

        // NOTE: The integer positions are in the center of the grid cells.
        tfluids_(vec3) pos;
        pos.x = static_cast<real>(i);
        pos.y = static_cast<real>(j);
        pos.z = static_cast<real>(k);

        // Velocity is in grids / second.
        tfluids_(vec3) vel;
        vel.x = u[IX(i, j, k, dims)];
        vel.y = v[IX(i, j, k, dims)];
        vel.z = two_dim ? static_cast<real>(0) : w[IX(i, j, k, dims)];

        // Backtrace a half step based upon current velocity at cell center.
        tfluids_(vec3) half_pos;
        tfluids_(vec3) displacement;
        displacement.x = (static_cast<real>(0.5) * ndt * vel.x);
        displacement.y = (static_cast<real>(0.5) * ndt * vel.y);
        displacement.z = (static_cast<real>(0.5) * ndt * vel.z);
        const bool hit_boundary_half =
            calcLineTrace(pos, displacement, dims, obs, half_pos);


        // Check the return value from calcLineTrace just in case.
        if (IsOutOfDomainReal(half_pos, dims) ||
            IsBlockedCellReal(obs, half_pos, dims)) {
          printf("Error: advectScalarRK2 - calcLineTrace returned an "
                 "invalid cell (half_pos).\n");
          exit(-1);
        }


        if (hit_boundary_half) {
          // We hit the boundary, then as per Bridson, we should clamp the
          // backwards trace. Note: if we treated this as a full euler step, we 
          // would have hit the same blocker because the line trace is linear.
          // TODO(tompson,kris): I'm pretty sure this is the best we could do
          // but I still worry about numerical stability.
          q_dst[IX(i, j, k, dims)] = sampleField(q_src, half_pos, obs, dims,
                                                 sample_into_geom);
          continue;
        }

        // Sample the velocity at this half step location.
        vel.x = sampleField(u, half_pos, obs, dims, true);
        vel.y = sampleField(v, half_pos, obs, dims, true);
        if (!two_dim) {
          vel.z = sampleField(w, half_pos, obs, dims, true);
        } else {
          vel.z = static_cast<real>(0);
        }

        // Do another line trace using this half position's velocity.
        tfluids_(vec3) back_pos;
        displacement.x = (ndt * vel.x);
        displacement.y = (ndt * vel.y);
        displacement.z = (ndt * vel.z);
        const bool hit_boundary = calcLineTrace(pos, displacement, dims, obs,
                                                back_pos);

        // Again, check the return value from calcLineTrace just in case.
        if (IsOutOfDomainReal(back_pos, dims) ||
            IsBlockedCellReal(obs, back_pos, dims)) {
          printf("Error: advectScalarRK2 - calcLineTrace returned an "
                 "invalid cell (back_pos).\n");
          exit(-1);
        }

        // Sample the value at the new position.
        q_dst[IX(i, j, k, dims)] = sampleField(q_src, back_pos, obs, dims,
                                               sample_into_geom);
      }
    }
  }
}

// Perform a self-advection of the input velocity field using a first order
// (Euler) step.
//
// @param w - OPTIONAL: For 2D systems this can be nullptr. If w is nullptr
// then w_dst must also be nullptr.
//
// TODO(tompson,kris): Refactor this so that the velocity and scalar functions
// don't have code duplication.
static void tfluids_(Main_advectVelEuler)(
    const real dt, const real* u, const real* v, const real* w,
    const real* obs, const Int3& dims, real* u_dst, real* v_dst, real* w_dst) {
  const real ndt = -dt;
  const bool two_dim = w == nullptr;
  if (!(two_dim == (w_dst == nullptr))) {
    printf("Error: advectVelEuler two_dim does not match 'w'.\n");
    exit(-1);
  }
  // TODO(tompson,kris): ALLOW FOR DYNAMIC OBSTACLES. Right now we are
  // assuming that obstacles have zero velocity
  int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
  for(k = 0; k < dims.z; k++) { 
    for(j = 0; j < dims.y; j++) { 
      for(i = 0; i < dims.x; i++) {
        if (IsBlockedCell(obs, i, j, k, dims)) {
          // Don't advect blocked cells.
          continue;
        }

        // NOTE: The integer positions are in the center of the grid cells.
        tfluids_(vec3) pos;
        pos.x = static_cast<real>(i);
        pos.y = static_cast<real>(j);
        pos.z = static_cast<real>(k);

        // Velocity is in grids / second.
        tfluids_(vec3) vel;
        vel.x = u[IX(i, j, k, dims)];
        vel.y = v[IX(i, j, k, dims)];
        vel.z = two_dim ? static_cast<real>(0) : w[IX(i, j, k, dims)];

        // Backtrace based upon current velocity at cell center.
        tfluids_(vec3) back_pos;
        tfluids_(vec3) displacement;
        displacement.x = (ndt * vel.x);
        displacement.y = (ndt * vel.y);
        displacement.z = (ndt * vel.z);
        // Step along the displacement vector. calcLineTrace will handle
        // boundary conditions for us. Note: it will terminate BEFORE the
        // boundary (i.e. the returned position is always valid).
        const bool hit_boundary = calcLineTrace(pos, displacement, dims, obs,
                                                back_pos);

        // Check the return value from calcLineTrace just in case.
        if (IsOutOfDomainReal(back_pos, dims) ||
            IsBlockedCellReal(obs, back_pos, dims)) {
          printf("Error: advectVelEuler - calcLineTrace returned an "
                 "invalid cell (back_pos).\n");
          exit(-1);
        }

        // Finally, sample the value at the new position.
        u_dst[IX(i, j, k, dims)] = sampleField(u, back_pos, obs, dims, true);
        v_dst[IX(i, j, k, dims)] = sampleField(v, back_pos, obs, dims, true);
        if (!two_dim) {
          w_dst[IX(i, j, k, dims)] = sampleField(w, back_pos, obs, dims, true);
        }
      }
    }
  }
}

// Second order (RK2) version of the self-advection function.
static void tfluids_(Main_advectVelRK2)(
    const real dt, const real* u, const real* v, const real* w,
    const real* obs, const Int3& dims, real* u_dst, real* v_dst, real* w_dst) {
  const real ndt = -dt;
  const bool two_dim = w == nullptr;
  if (!(two_dim == (w_dst == nullptr))) {
    printf("Error: advectVelRK2 two_dim does not match 'w'.\n");
    exit(-1);
  }
  // TODO(tompson,kris): ALLOW FOR DYNAMIC OBSTACLES. Right now we are
  // assuming that obstacles have zero velocity
  int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
  for(k = 0; k < dims.z; k++) { 
    for(j = 0; j < dims.y; j++) { 
      for(i = 0; i < dims.x; i++) {
        if (IsBlockedCell(obs, i, j, k, dims)) {
          // Don't advect blocked cells.
          continue;
        }

        // NOTE: The integer positions are in the center of the grid cells.
        tfluids_(vec3) pos;
        pos.x = static_cast<real>(i);
        pos.y = static_cast<real>(j);
        pos.z = static_cast<real>(k);

        // Velocity is in grids / second.
        tfluids_(vec3) vel;
        vel.x = u[IX(i, j, k, dims)];
        vel.y = v[IX(i, j, k, dims)];
        vel.z = two_dim ? static_cast<real>(0) : w[IX(i, j, k, dims)];

        // Backtrace a half step based upon current velocity at cell center.
        tfluids_(vec3) half_pos;
        tfluids_(vec3) displacement;
        displacement.x = (static_cast<real>(0.5) * ndt * vel.x);
        displacement.y = (static_cast<real>(0.5) * ndt * vel.y);
        displacement.z = (static_cast<real>(0.5) * ndt * vel.z);
        const bool hit_boundary_half =
            calcLineTrace(pos, displacement, dims, obs, half_pos);

        // Check the return value from calcLineTrace just in case.
        if (IsOutOfDomainReal(half_pos, dims) ||
            IsBlockedCellReal(obs, half_pos, dims)) {
          printf("Error: advectVelRK2 - calcLineTrace returned an "
                 "invalid cell (half_pos).\n");
          exit(-1);
        }

        if (hit_boundary_half) {
          u_dst[IX(i, j, k, dims)] = sampleField(u, half_pos, obs, dims, true);
          v_dst[IX(i, j, k, dims)] = sampleField(v, half_pos, obs, dims, true);
          if (!two_dim) {
            if (w_dst == nullptr) {
              printf("ERROR: null w component found!\n");
              exit(-1);
            }
            w_dst[IX(i, j, k, dims)] = sampleField(w, half_pos, obs, dims,
                                                   true);
          }
          continue;
        }

        // Sample the velocity at this half step location.
        vel.x = sampleField(u, half_pos, obs, dims, true);
        vel.y = sampleField(v, half_pos, obs, dims, true);
        if (!two_dim) {
          vel.z = sampleField(w, half_pos, obs, dims, true);
        } else {
          vel.z = static_cast<real>(0);
        }

        // Do another line trace using this half position's velocity.
        tfluids_(vec3) back_pos;
        displacement.x = (ndt * vel.x);
        displacement.y = (ndt * vel.y);
        displacement.z = (ndt * vel.z);
        const bool hit_boundary = calcLineTrace(pos, displacement, dims, obs,
                                                back_pos);

        // Again, check the return value from calcLineTrace just in case.
        if (IsOutOfDomainReal(back_pos, dims) ||
            IsBlockedCellReal(obs, back_pos, dims)) {
          printf("Error: advectVelRK2 - calcLineTrace returned an "
                 "invalid cell (back_pos).\n");
          exit(-1);
        }

        // Sample the value at the new position.
        u_dst[IX(i, j, k, dims)] = sampleField(u, back_pos, obs, dims, true);
        v_dst[IX(i, j, k, dims)] = sampleField(v, back_pos, obs, dims, true);
        if (!two_dim) {
          if (w_dst == nullptr) {
            printf("ERROR: null w component found!\n");
            exit(-1);
          }
          w_dst[IX(i, j, k, dims)] = sampleField(w, back_pos, obs, dims, true);
        }
      }
    }
  }
}

// *****************************************************************************
// LUA MAIN ENTRY POINT FUNCTIONS
// *****************************************************************************

static int tfluids_(Main_advectScalar)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* p_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  const std::string method = static_cast<std::string>(lua_tostring(L, 6));
  const bool sample_into_geom = static_cast<bool>(lua_toboolean(L, 7));

  bool two_dim = u->size[0] == 2;
  const int32_t xdim = static_cast<int32_t>(geom->size[2]);
  const int32_t ydim = static_cast<int32_t>(geom->size[1]);
  const int32_t zdim = static_cast<int32_t>(geom->size[0]);

  // Get pointers to the tensor data.
  const real* p_data = THTensor_(data)(p);
  const real* u_data = THTensor_(data)(u);
  const real* geom_data = THTensor_(data)(geom);
  real* p_dst_data = THTensor_(data)(p_dst);

  const real* ux_data = &u_data[0 * xdim * ydim * zdim];
  const real* uy_data = &u_data[1 * xdim * ydim * zdim];
  const real* uz_data = two_dim ? nullptr : &u_data[2 * xdim * ydim * zdim];

  // Finally, call the advection routine.
  Int3 dim;
  dim.x = xdim;
  dim.y = ydim;
  dim.z = zdim;
  if (method == "rk2") { 
    tfluids_(Main_advectScalarRK2)(dt, p_data, ux_data, uy_data, uz_data,
                                   geom_data, dim, p_dst_data,
                                   sample_into_geom);
  } else if (method == "euler") {
    tfluids_(Main_advectScalarEuler)(dt, p_data, ux_data, uy_data, uz_data,
                                     geom_data, dim, p_dst_data,
                                     sample_into_geom);
  } else if (method == "maccormack") {
    luaL_error(L, "maccormack not yet implemented.");
  } else {
    luaL_error(L, "Invalid advection method.");
  }
  
  return 0;  // Recall: number of return values on the lua stack.
}

static int tfluids_(Main_advectVel)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  real dt = static_cast<real>(lua_tonumber(L, 1));
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* u_dst =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  const std::string method = static_cast<std::string>(lua_tostring(L, 5));

  bool two_dim = u->size[0] == 2;
  const int32_t xdim = static_cast<int32_t>(geom->size[2]);
  const int32_t ydim = static_cast<int32_t>(geom->size[1]);
  const int32_t zdim = static_cast<int32_t>(geom->size[0]);

  // Get pointers to the tensor data.
  const real* u_data = THTensor_(data)(u);
  const real* geom_data = THTensor_(data)(geom);
  real* u_dst_data = THTensor_(data)(u_dst);

  const real* ux_data = &u_data[0 * xdim * ydim * zdim];
  const real* uy_data = &u_data[1 * xdim * ydim * zdim];
  const real* uz_data = two_dim ? nullptr : &u_data[2 * xdim * ydim * zdim];

  real* ux_dst_data = &u_dst_data[0 * xdim * ydim * zdim];
  real* uy_dst_data = &u_dst_data[1 * xdim * ydim * zdim];
  real* uz_dst_data = two_dim ? nullptr : &u_dst_data[2 * xdim * ydim * zdim];

  // Finally, call the advection routine.
  Int3 dim;   
  dim.x = xdim;
  dim.y = ydim;
  dim.z = zdim;
  if (method == "rk2") {
    tfluids_(Main_advectVelRK2)(dt, ux_data, uy_data, uz_data,
                                geom_data, dim, ux_dst_data, uy_dst_data,
                                uz_dst_data);
  } else if (method == "euler") {
    tfluids_(Main_advectVelEuler)(dt, ux_data, uy_data, uz_data,
                                  geom_data, dim, ux_dst_data, uy_dst_data,
                                  uz_dst_data);
  } else if (method == "maccormack") {
    luaL_error(L, "maccormack not yet implemented.");
  } else {
    luaL_error(L, "Invalid advection method.");
  }
 
  return 0;  // Number of return values on the lua stack.
}

static int tfluids_(Main_vorticityConfinement)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack. We also treat 2D as 3D (with depth = 1) and
  // no 'w' component for velocity.
  const real dt = static_cast<real>(lua_tonumber(L, 1));
  const real scale = static_cast<real>(lua_tonumber(L, 2));
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* curl =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));
  THTensor* mag_curl =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));

  if (u->nDimension != 4 || geom->nDimension != 3 ||
      mag_curl->nDimension != 3) {
    luaL_error(L, "Incorrect input sizes.");
  }

  bool two_dim = u->size[0] == 2;
  const int32_t xdim = static_cast<int32_t>(geom->size[2]);
  const int32_t ydim = static_cast<int32_t>(geom->size[1]);
  const int32_t zdim = static_cast<int32_t>(geom->size[0]);

  if (two_dim && curl->nDimension != 3) {
     luaL_error(L, "Bad curl size.");
  }
  if (!two_dim && curl->nDimension != 4) {
     luaL_error(L, "Bad curl size.");
  }

  if (two_dim && zdim != 1) {
    luaL_error(L, "Incorrect input sizes.");
  }

  if (!two_dim && u->size[0] != 3) {
    luaL_error(L, "Incorrect input sizes.");
  }

  // Get pointers to the tensor data.
  real* u_data = THTensor_(data)(u);
  const real* geom_data = THTensor_(data)(geom);
  real* curl_data = THTensor_(data)(curl);
  real* mag_curl_data = THTensor_(data)(mag_curl);

  real* ux_data = &u_data[0 * xdim * ydim * zdim];
  real* uy_data = &u_data[1 * xdim * ydim * zdim];
  real* uz_data = two_dim ? nullptr : &u_data[2 * xdim * ydim * zdim];

  real* curl_u = &curl_data[0 * xdim * ydim * zdim];
  real* curl_v = two_dim ? nullptr : &curl_data[1 * xdim * ydim * zdim];
  real* curl_w = two_dim ? nullptr : &curl_data[2 * xdim * ydim * zdim];

  // Finally, call the vorticity confinement routine.
  Int3 dim;
  dim.x = xdim;
  dim.y = ydim;
  dim.z = zdim;
  tfluids_(Main_vorticityConfinement)(dt, scale, ux_data, uy_data, uz_data,
                                      geom_data, dim, curl_u, curl_v, curl_w,
                                      mag_curl_data);
 
  return 0;  // Number of return values on the lua stack.
}

static int tfluids_(Main_averageBorderCells)(lua_State *L) {
  THTensor* in =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* out =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  if (in->nDimension != 4 || geom->nDimension != 3) {
    luaL_error(L, "Input tensor should be 4D and geom should be 3D");
  }
  const int32_t nchan = in->size[0];
  const int32_t zdim = in->size[1];
  const int32_t ydim = in->size[2];
  const int32_t xdim = in->size[3];

  const bool two_dim = zdim == 1;

  real* in_data = THTensor_(data)(in);
  real* out_data = THTensor_(data)(out);
  real* geom_data = THTensor_(data)(geom);

  // Average the pixels in a neighborhood. We want the average to be
  // symmetric and so should be all surrounding neighbors.
  // Store the result in out.
  int32_t c, z, y, x, zoff, yoff, xoff;
#pragma omp parallel for private(c,z,y,x,zoff,yoff,xoff) collapse(4)
  for (c = 0; c < nchan; c++) {
    for (z = 0; z < zdim; z++) {
      for (y = 0; y < ydim; y++) {
        for (x = 0; x < xdim; x++) {
          const int32_t idst = (c * xdim * ydim * zdim) + (z * xdim * ydim) +
              (y * xdim) + x;
          if ((!two_dim && (z == 0 || z == (zdim - 1))) ||
              y == 0 || y == (ydim - 1) ||
              x == 0 || x == (xdim - 1)) {
            // We're a border pixel.
            // TODO(tompson): This is an O(n^3) iteration to find a small
            // sub-set of the pixels. Fix it.
            real mean = 0;
            int32_t count = 0;
            // Count and sum the number of non-geometry and in-boundary pixels.
            for (zoff = z - 1; zoff <= z + 1; zoff++) {
              for (yoff = y - 1; yoff <= y + 1; yoff++) {
                for (xoff = x - 1; xoff <= x + 1; xoff++) {
                  if (zoff >= 0 && zoff < zdim &&
                      yoff >= 0 && yoff < ydim &&
                      xoff >= 0 && xoff < xdim) {
                    // The neighbor is on the image.
                    const int32_t ipix = (zoff * xdim * ydim) + (yoff * xdim) +
                        xoff;
                    if (geom_data[ipix] < static_cast<real>(1e-6)) {
                      // The neighbor is NOT geometry.
                      count++;
                      mean += in_data[(c * xdim * ydim * zdim) + ipix];
                    }
                  }
                }
              }
            }
            if (count > 0) {
              out_data[idst] = mean / static_cast<real>(count);
            } else {
              // No non-geom pixels found. Just copy over result.
              out_data[idst] = in_data[idst];
            }
          } else {
            // Internal cell, just copy the result.
            out_data[idst] = in_data[idst];
          }
        }
      }
    }
  }

  return 0;
}

static int tfluids_(Main_setObstacleBcs)(lua_State *L) {
  THTensor* U =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
 
  if (U->nDimension != 4 || geom->nDimension != 3) {
    luaL_error(L, "Input U should be 4D and geom should be 3D");
  }
  const bool two_dim = U->size[0] == 2;
  if (!two_dim && U->size[0] != 3) {
    luaL_error(L, "Unexpected size 1 for the U tensor.");
  }
  const int32_t zdim = U->size[1];
  const int32_t ydim = U->size[2];
  const int32_t xdim = U->size[3];
  Int3 dims;
  dims.x = xdim;
  dims.y = ydim;
  dims.z = zdim;

  if (geom->size[0] != zdim || geom->size[1] != ydim || geom->size[2] != xdim) {
    luaL_error(L, "Inconsistent geometry size.");
  }

  if (two_dim && zdim != 1) {
    luaL_error(L, "2D input does not have zdim == 1.");
  }

  real* u_data = THTensor_(data)(U);
  real* ux_data = &u_data[0];
  real* uy_data = &u_data[xdim * ydim * zdim];
  real* uz_data;
  if (!two_dim) {
    uz_data = &u_data[2 * xdim * ydim * zdim];
  }
  const real* geom_data = THTensor_(data)(geom);

  int32_t k, j, i;
  // Now accumulate velocity into geometry cells so that the face velocity
  //components are zero.
#pragma omp parallel for private(k,j,i) collapse(3)
  for (k = 0; k < zdim; k++) {  
    for (j = 0; j < ydim; j++) { 
      for (i = 0; i < xdim; i++) {
        if (IsBlockedCell(geom_data, i, j, k, dims)) {
          // Ignore fluid cells.

          // Otherwise this is a geometry cell. Zero the velocity component then
          // accumulate adjacent fluid cell velocities so that the boundary
          // velocity is zero.
          ux_data[IX(i, j, k, dims)] = 0;
          uy_data[IX(i, j, k, dims)] = 0;
          if (!two_dim) {
            uz_data[IX(i, j, k, dims)] = 0;
          }
  
          // Spread adjacent fluid velocities so that the face velocity values
          // are zero along the normal of the face.
  
          // Look -x..
          if (i > 0 && !IsBlockedCell(geom_data, i - 1, j, k, dims)) {
            ux_data[IX(i, j, k, dims)] -= ux_data[IX(i - 1, j, k, dims)];
          }
          // Look +x.
          if (i < (xdim - 1) && !IsBlockedCell(geom_data, i + 1, j, k, dims)) {
            ux_data[IX(i, j, k, dims)] -= ux_data[IX(i + 1, j, k, dims)];
          }
          // Look -y..
          if (j > 0 && !IsBlockedCell(geom_data, i, j - 1, k, dims)) {
            uy_data[IX(i, j, k, dims)] -= uy_data[IX(i, j - 1, k, dims)];
          }
          // Look +y.
          if (j < (ydim - 1) && !IsBlockedCell(geom_data, i, j + 1, k, dims)) {
            uy_data[IX(i, j, k, dims)] -= uy_data[IX(i, j + 1, k, dims)];
          }
          if (!two_dim) {
            // Look -z..
            if (k > 0 && !IsBlockedCell(geom_data, i, j, k - 1, dims)) {
              uz_data[IX(i, j, k, dims)] -= uz_data[IX(i, j, k - 1, dims)];
            }
            // Look +z.
            if (k < (zdim - 1) &&
                !IsBlockedCell(geom_data, i, j, k + 1, dims)) {
              uz_data[IX(i, j, k, dims)] -= uz_data[IX(i, j, k + 1, dims)];
            }
          }
        }
      }
    }
  }
  return 0;
}

// Expose the getInterpValue to the call to the lua stack for debugging.
static int tfluids_(Main_interpField)(lua_State *L) {
  THTensor* field =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* pos =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool sample_into_geom = static_cast<bool>(lua_toboolean(L, 4));

  if (field->nDimension != 3 || geom->nDimension != 3) {
    luaL_error(L, "Input field and geom should be 3D.");
  }
  if (pos->nDimension != 1 || pos->size[0] != 3) {
    luaL_error(L, "pos should be 1D and size 3.");
  }
  const int32_t zdim = field->size[0];
  const int32_t ydim = field->size[1];
  const int32_t xdim = field->size[2];
  Int3 dims;
  dims.x = xdim;
  dims.y = ydim;
  dims.z = zdim;

  const real* field_data = THTensor_(data)(field);
  const real* geom_data = THTensor_(data)(geom);
  const real* pos_data = THTensor_(data)(pos);

  tfluids_(vec3) interp_pos;
  interp_pos.x = pos_data[0];
  interp_pos.y = pos_data[1];
  interp_pos.z = pos_data[2];

  const real ret_val = tfluids_(Main_GetInterpValue)(
    field_data, geom_data, interp_pos, dims, sample_into_geom);

  lua_pushnumber(L, static_cast<double>(ret_val));
  return 1;
}

static int tfluids_(Main_drawVelocityField)(lua_State *L) {
#ifdef BUILD_GL_FUNCS
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));

  const bool flip_y = static_cast<bool>(lua_toboolean(L, 2));

  if (u->nDimension != 5) {
    luaL_error(L, "Input vector field should be 5D.");
  }
  const int32_t nbatch = u->size[0];
  const int32_t nchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];
  const bool two_dim = nchan == 2;
  if (two_dim && zdim != 1) {
    luaL_error(L, "Unexpected zdim for 2D vector field.");
  }

  const real* u_data = THTensor_(data)(u);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glBegin(GL_LINES);
  for (int32_t b = 0; b < nbatch; b++) {
    for (int32_t z = 0; z < zdim; z++) {
      for (int32_t y = 0; y < ydim; y++) {
        for (int32_t x = 0; x < xdim; x++) {
          const int32_t ux_index = b * nchan * zdim * ydim * xdim +
              z * ydim * xdim + y * xdim + x;
          real ux = u_data[ux_index];
          real uy = u_data[ux_index + xdim * ydim * zdim];
          real uz = two_dim ? static_cast<real>(0) :
              u_data[ux_index + 2 * xdim * ydim * zdim];
          // Velocity is in grids / second. But we need coordinates in [0, 1].
          ux = ux / static_cast<real>(xdim - 1);
          uy = uy / static_cast<real>(ydim - 1);
          if (!two_dim) {
            uz = uz / static_cast<real>(zdim - 1);
          }
          // Same for position.
          real px = static_cast<real>(x) / static_cast<real>(xdim - 1);
          real py = static_cast<real>(y) / static_cast<real>(ydim - 1);
          real pz = two_dim ? static_cast<real>(0) :
              (static_cast<real>(z) / static_cast<real>(zdim - 1));
          py = flip_y ? py : static_cast<real>(1) - py;
          uy = flip_y ? -uy : uy;
          glColor4f(0.7f, 0.0f, 0.0f, 1.0f);
          glVertex3f(static_cast<float>(px),
                     static_cast<float>(py),
                     static_cast<float>(pz));
          glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
          glVertex3f(static_cast<float>(px + ux),
                     static_cast<float>(py - uy),
                     static_cast<float>(pz + uz));
        }
      }
    }
  }
  glEnd();
#else
  luaL_error(L, "tfluids compiled without preprocessor def BUILD_GL_FUNCS.");
#endif
  return 0;
}

static int tfluids_(Main_loadTensorTexture)(lua_State *L) {
#ifdef BUILD_GL_FUNCS
  THTensor* im_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  if (im_tensor->nDimension != 2 && im_tensor->nDimension != 3) {
    luaL_error(L, "Input should be 2D or 3D.");
  }
  const int32_t tex_id = static_cast<int32_t>(luaL_checkinteger(L, 2));
  if (!lua_isboolean(L, 3)) {
    luaL_error(L, "3rd argument to loadTensorTexture should be boolean.");
  }
  const bool filter = lua_toboolean(L, 3);
  if (!lua_isboolean(L, 4)) {
    luaL_error(L, "4rd argument to loadTensorTexture should be boolean.");
  }
  const bool flip_y = lua_toboolean(L, 4);

  const bool grey = im_tensor->nDimension == 2;
  const int32_t nchan = grey ? 1 : im_tensor->size[0];
  const int32_t h = grey ? im_tensor->size[0] : im_tensor->size[1];
  const int32_t w = grey ? im_tensor->size[1] : im_tensor->size[2];

  if (nchan != 1 && nchan != 3) {
    luaL_error(L, "Only 3 or 1 channels is supported.");
  }

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, tex_id);

  if (filter) {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  } else {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  const real* im_tensor_data = THTensor_(data)(im_tensor);

  // We need to either: a) swizzle the RGB data, b) convert from double to float
  // or c) convert to RGB greyscale for single channel textures). For c) we
  // could use a swizzle mask, but this complicates alpha blending, so it's
  // easier to just convert always at the cost of a copy (which is parallel and
  // fast).

  std::unique_ptr<float[]> fdata(new float[h * w * 4]);
  int32_t c, u, v;
#pragma omp parallel for private(v, u, c) collapse(3)
  for (v = 0; v < h; v++) {
    for (u = 0; u < w; u++) {
      for (c = 0; c < 4; c++) {
        if (c == 3) {
          // OpenMP requires perfectly nested loops, so we need to include the
          // alpha chan set like this.
          fdata[v * 4 * w + u * 4 + c] = 1.0f;
        } else {
          const int32_t csrc = (c < nchan) ? c : 0;
          const int32_t vsrc = flip_y ? (h - v - 1) : v;
          fdata[v * 4 * w + u * 4 + c] =
              static_cast<float>(im_tensor_data[csrc * w * h + vsrc * w + u]);
        }
      }
    }
  }

  const GLint level = 0;
  const GLint internalformat = GL_RGBA32F;
  const GLint border = 0;
  const GLenum format = GL_RGBA;
  const GLenum type = GL_FLOAT;
  glTexImage2D(GL_TEXTURE_2D, level, internalformat, w, h, border,
               format, type, fdata.get());
#else
  luaL_error(L, "tfluids compiled without preprocessor def BUILD_GL_FUNCS.");
#endif
  return 0;
}

static inline void tfluids_(Main_calcVelocityUpdateAlongDim)(
    real* delta_u, const real* p, const real* geom, const int32_t* pos,
    const int32_t* size, const int32_t dim, const bool match_manta) {
  const int32_t uslice = dim * size[0] * size[1] * size[2];

  // This is ugly, but make an Int3 dims container for our IX and IsBlockedCell
  // calls.
  Int3 dims;
  dims.x = size[0];
  dims.y = size[1];
  dims.z = size[2];

  if (IsBlockedCell(geom, pos[0], pos[1], pos[2], dims)) {
    delta_u[uslice + IX(pos[0], pos[1], pos[2], dims)] = 0;
    return;
  }

  int32_t pos_p[3] = {pos[0], pos[1], pos[2]};
  pos_p[dim] += 1;   
  int32_t pos_n[3] = {pos[0], pos[1], pos[2]};
  pos_n[dim] -= 1;

  // First annoying special case that happens on the border because of our
  // conversion to central velocities and because manta does not handle this
  // case properly.
  if (pos[dim] == 0 && match_manta) {
    if (IsBlockedCell(geom, pos_p[0], pos_p[1], pos_p[2], dims) &&
        !IsBlockedCell(geom, pos[0], pos[1], pos[2], dims)) {
      delta_u[uslice + IX(pos[0], pos[1], pos[2], dims)] =
          p[IX(pos[0], pos[1], pos[2], dims)] * static_cast<real>(0.5);
    } else {
      delta_u[uslice + IX(pos[0], pos[1], pos[2], dims)] =
          p[IX(pos_p[0], pos_p[1], pos_p[2], dims)] * static_cast<real>(0.5);
    }
    return;
  }

  // This function will perform the conditional partial derivative to calculate
  // the velocity update along a particular dimension.

  // Look at the neighbor to the right (pos) and to the left (neg).
  bool geomPos = false;
  bool geomNeg = false;
  if (pos[dim] == 0) {
    geomNeg = true;  // Treat going off the fluid as geometry.
  }
  if (pos[dim] == size[dim] - 1) {
    geomPos = true;  // Treat going off the fluid as geometry. 
  }
  if (pos[dim] > 0) {
    geomNeg = IsBlockedCell(geom, pos_n[0], pos_n[1], pos_n[2], dims);
  }
  if (pos[dim] < size[dim] - 1) {
    geomPos = IsBlockedCell(geom, pos_p[0], pos_p[1], pos_p[2], dims); 
  }

  // NOTE: The 0.5 below needs some explanation. We are exactly
  // mimicking CorrectVelocity() from
  // manta/source/pluging/pressure.cpp. In this function, all
  // updates are single sided, but they are done to the MAC cell
  // edges. When we convert to centered velocities, we therefore add
  // a * 0.5 term because we take the average.
  const real single_sided_gain = match_manta ? static_cast<real>(0.5) :
      static_cast<real>(1);

  if (geomPos and geomNeg) {
    // There are 3 cases:
    // A) Cell is on the left border and has a right geom neighbor.
    // B) Cell is on the right border and has a left geom neighbor.
    // C) Cell has a right AND left geom neighbor.
    // In any of these cases the velocity should not receive a
    // pressure gradient (nowhere for the pressure to diffuse.
    delta_u[uslice + IX(pos[0], pos[1], pos[2], dims)] = 0;
  } else if (geomPos) {
    // There are 2 cases:
    // A) Cell is on the right border and there's fluid to the left.
    // B) Cell is internal but there is geom to the right.
    // In this case we need to do a single sided diff to the left.
    delta_u[uslice + IX(pos[0], pos[1], pos[2], dims)] =
        (p[IX(pos[0], pos[1], pos[2], dims)] -
         p[IX(pos_n[0], pos_n[1], pos_n[2], dims)]) * single_sided_gain;
  } else if (geomNeg) {
    // There are 2 cases:
    // A) Cell is on the left border and there's fluid to the right.
    // B) Cell is internal but there is geom to the left.
    // In this case we need to do a single sided diff to the right.
    delta_u[uslice + IX(pos[0], pos[1], pos[2], dims)] =
        (p[IX(pos_p[0], pos_p[1], pos_p[2], dims)] -
         p[IX(pos[0], pos[1], pos[2], dims)]) * single_sided_gain;
  } else {
    // The pixel is internal (not on border) with no geom neighbours.
    // Do a central diff.
    delta_u[uslice + IX(pos[0], pos[1], pos[2], dims)] =
        (p[IX(pos_p[0], pos_p[1], pos_p[2], dims)] -
         p[IX(pos_n[0], pos_n[1], pos_n[2], dims)]) * static_cast<real>(0.5);
  }
}

static int tfluids_(Main_calcVelocityUpdate)(lua_State *L) {
  THTensor* delta_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool match_manta = static_cast<bool>(lua_toboolean(L, 4));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (delta_u->nDimension != 5 || p->nDimension != 4 || geom->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions. Expect delta_u: 5, p: 4, geom: 4.");
  }
  const int32_t nbatch = delta_u->size[0];
  const int32_t nuchan = delta_u->size[1];
  const int32_t zdim = delta_u->size[2];
  const int32_t ydim = delta_u->size[3];
  const int32_t xdim = delta_u->size[4];

  real* delta_u_data = THTensor_(data)(delta_u);
  const real* geom_data = THTensor_(data)(geom);
  const real* p_data = THTensor_(data)(p);

  int32_t b, c, z, y, x;
#pragma omp parallel for private(b, c, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (c = 0; c < nuchan; c++) {
      for (z = 0; z < zdim; z++) {
        for (y = 0; y < ydim; y++) {
          for (x = 0; x < xdim; x++) {
            real* cur_delta_u = &delta_u_data[b * nuchan * zdim * ydim * xdim];
            const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
            const real* cur_p = &p_data[b * zdim * ydim * xdim];
            const int32_t pos[3] = {x, y, z};
            const int32_t size[3] = {xdim, ydim, zdim};

            tfluids_(Main_calcVelocityUpdateAlongDim)(
                cur_delta_u, cur_p, cur_geom, pos, size, c, match_manta);
          }
        }
      }
    }
  }
  return 0;
}

static inline void tfluids_(Main_calcVelocityUpdateAlongDimBackward)(
    real* grad_p, const real* p, const real* geom, const real* grad_output, 
    const int32_t* pos, const int32_t* size, const int32_t dim,
    const bool match_manta) {
  const int32_t uslice = dim * size[0] * size[1] * size[2];

  Int3 dims;
  dims.x = size[0];
  dims.y = size[1];
  dims.z = size[2];

  if (IsBlockedCell(geom, pos[0], pos[1], pos[2], dims)) {
    // No gradient contribution from blocked cells (since U(blocked) == 0).
    return;
  }

  int32_t pos_p[3] = {pos[0], pos[1], pos[2]};
  pos_p[dim] += 1;   
  int32_t pos_n[3] = {pos[0], pos[1], pos[2]};
  pos_n[dim] -= 1;

  if (pos[dim] == 0 && match_manta) {
    if (IsBlockedCell(geom, pos_p[0], pos_p[1], pos_p[2], dims) &&
        !IsBlockedCell(geom, pos[0], pos[1], pos[2], dims)) {
#pragma omp atomic
      grad_p[IX(pos[0], pos[1], pos[2], dims)] += static_cast<real>(0.5) * 
          grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)];
    } else {
#pragma omp atomic
      grad_p[IX(pos_p[0], pos_p[1], pos_p[2], dims)] += static_cast<real>(0.5) *
          grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)];
    }
    return;
  }

  bool geomPos = false;
  bool geomNeg = false;
  if (pos[dim] == 0) {
    geomNeg = true;
  }
  if (pos[dim] == size[dim] - 1) {
    geomPos = true;
  }
  if (pos[dim] > 0) {
    geomNeg = IsBlockedCell(geom, pos_n[0], pos_n[1], pos_n[2], dims);
  }
  if (pos[dim] < size[dim] - 1) {
    geomPos = IsBlockedCell(geom, pos_p[0], pos_p[1], pos_p[2], dims); 
  }

  const real single_sided_gain = match_manta ? static_cast<real>(0.5) :
      static_cast<real>(1);

  if (geomPos and geomNeg) {
    // Output velocity update is zero.
    // --> No gradient contribution from this case (since delta_u == 0).
  } else if (geomPos) {
    // Single sided diff to the left --> Spread the gradient contribution.
#pragma omp atomic
    grad_p[IX(pos[0], pos[1], pos[2], dims)] += single_sided_gain *
        grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)];
#pragma omp atomic
    grad_p[IX(pos_n[0], pos_n[1], pos_n[2], dims)] -= single_sided_gain *
        grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)];
  } else if (geomNeg) {
    // Single sided diff to the right --> Spread the gradient contribution.
#pragma omp atomic
    grad_p[IX(pos_p[0], pos_p[1], pos_p[2], dims)] += single_sided_gain *
        grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)];
#pragma omp atomic
    grad_p[IX(pos[0], pos[1], pos[2], dims)] -= single_sided_gain *
        grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)];
  } else {
    // Central diff --> Spread the gradient contribution.
#pragma omp atomic
    grad_p[IX(pos_p[0], pos_p[1], pos_p[2], dims)] += (static_cast<real>(0.5) *
        grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)]);
#pragma omp atomic
    grad_p[IX(pos_n[0], pos_n[1], pos_n[2], dims)] -= (static_cast<real>(0.5) *
        grad_output[uslice + IX(pos[0], pos[1], pos[2], dims)]);
  }
}

static int tfluids_(Main_calcVelocityUpdateBackward)(lua_State *L) {
  THTensor* grad_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  const bool match_manta = static_cast<bool>(lua_toboolean(L, 5));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (grad_output->nDimension != 5 || p->nDimension != 4 ||
      geom->nDimension != 4 || grad_p->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int32_t nbatch = grad_p->size[0];
  const int32_t zdim = grad_p->size[1];
  const int32_t ydim = grad_p->size[2];
  const int32_t xdim = grad_p->size[3];
  const int32_t nuchan = grad_output->size[1];

  real* grad_p_data = THTensor_(data)(grad_p);
  const real* geom_data = THTensor_(data)(geom);
  const real* p_data = THTensor_(data)(p);
  const real* grad_output_data = THTensor_(data)(grad_output);

  // We will be accumulating gradient contributions into the grad_p tensor, so
  // we first need to zero it.
  THTensor_(zero)(grad_p);

  // TODO(tompson): I have implemented the following function as a scatter
  // operation. However this requires the use of #pragma omp atomic everywhere.
  // Instead re-write the inner loop code to perform a gather op to avoid the
  // atomic locks.
  int32_t b, c, z, y, x;
#pragma omp parallel for private(b, c, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (c = 0; c < nuchan; c++) {
      for (z = 0; z < zdim; z++) {
        for (y = 0; y < ydim; y++) {
          for (x = 0; x < xdim; x++) {
            const real* cur_grad_output =
                &grad_output_data[b * nuchan * zdim * ydim * xdim];
            const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
            const real* cur_p = &p_data[b * zdim * ydim * xdim];
            real* cur_grad_p = &grad_p_data[b * zdim * ydim * xdim];
            const int32_t pos[3] = {x, y, z};
            const int32_t size[3] = {xdim, ydim, zdim};

            tfluids_(Main_calcVelocityUpdateAlongDimBackward)(
                cur_grad_p, cur_p, cur_geom, cur_grad_output, pos, size, c,
                match_manta);
          }
        }
      }
    }
  }
  return 0;
}

static inline void tfluids_(Main_calcVelocityDivergenceCell)(
    const real* u, const real* geom, real* u_div, const int32_t* pos,
    const int32_t* size, const int32_t nuchan) {
  Int3 dims;
  dims.x = size[0];
  dims.y = size[1];
  dims.z = size[2];

  // Zero the output divergence.
  u_div[IX(pos[0], pos[1], pos[2], dims)] = 0;

  if (IsBlockedCell(geom, pos[0], pos[1], pos[2], dims)) {
    // Divergence INSIDE geometry is zero always, or we don't try to minimize it
    // during training.
    return;
  }
 
  // Now calculate the partial derivatives in each dimension.
  for (int32_t dim = 0; dim < nuchan; dim ++) {
    const int32_t uslice = dim * size[0] * size[1] * size[2];

    int32_t pos_p[3] = {pos[0], pos[1], pos[2]};
    pos_p[dim] += 1;   
    int32_t pos_n[3] = {pos[0], pos[1], pos[2]};
    pos_n[dim] -= 1;

    // Look at the neighbor to the right (pos) and to the left (neg).
    bool geomPos = false;
    bool geomNeg = false;
    if (pos[dim] == 0) {
      geomNeg = true;  // Treat going off the fluid as geometry.
    }
    if (pos[dim] == size[dim] - 1) {
      geomPos = true;  // Treat going off the fluid as geometry. 
    }
    if (pos[dim] > 0) {
      geomNeg = IsBlockedCell(geom, pos_n[0], pos_n[1], pos_n[2], dims);
    }
    if (pos[dim] < size[dim] - 1) {
      geomPos = IsBlockedCell(geom, pos_p[0], pos_p[1], pos_p[2], dims); 
    }

    if (geomPos and geomNeg) {
      // We are bordered by two geometry voxels OR one voxel and the border.
      // Treat the current partial derivative w.r.t. dim as 0.
      continue;
    } else if (geomPos) {
      // There are 2 cases:
      // A) Cell is on the right border and there's fluid to the left.
      // B) Cell is internal but there is geom to the right.
      // In this case we need to do a single sided diff to the left.
      u_div[IX(pos[0], pos[1], pos[2], dims)] +=
          (u[uslice + IX(pos[0], pos[1], pos[2], dims)] -
           u[uslice + IX(pos_n[0], pos_n[1], pos_n[2], dims)]);
    } else if (geomNeg) {
      // There are 2 cases:
      // A) Cell is on the left border and there's fluid to the right.
      // B) Cell is internal but there is geom to the left.
      // In this case we need to do a single sided diff to the right.
      u_div[IX(pos[0], pos[1], pos[2], dims)] +=
          (u[uslice + IX(pos_p[0], pos_p[1], pos_p[2], dims)] -
           u[uslice + IX(pos[0], pos[1], pos[2], dims)]);
    } else {
      // The pixel is internal (not on border) with no geom neighbours.
      // Do a central diff.
      u_div[IX(pos[0], pos[1], pos[2], dims)] += static_cast<real>(0.5) *
          (u[uslice + IX(pos_p[0], pos_p[1], pos_p[2], dims)] -
           u[uslice + IX(pos_n[0], pos_n[1], pos_n[2], dims)]);
    }
  }
}

static int tfluids_(Main_calcVelocityDivergence)(lua_State *L) {
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* u_div =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (u->nDimension != 5 || u_div->nDimension != 4 || geom->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int32_t nbatch = u->size[0];
  const int32_t nuchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];

  real* u_div_data = THTensor_(data)(u_div);
  const real* geom_data = THTensor_(data)(geom);
  const real* u_data = THTensor_(data)(u);

  int32_t b, z, y, x;
#pragma omp parallel for private(b, z, y, x) collapse(4)
  for (b = 0; b < nbatch; b++) {
    for (z = 0; z < zdim; z++) {
      for (y = 0; y < ydim; y++) {
        for (x = 0; x < xdim; x++) {
          const real* cur_u = &u_data[b * nuchan * zdim * ydim * xdim];
          const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
          real* cur_u_div = &u_div_data[b * zdim * ydim * xdim];
          const int32_t pos[3] = {x, y, z};
          const int32_t size[3] = {xdim, ydim, zdim};

          tfluids_(Main_calcVelocityDivergenceCell)(
              cur_u, cur_geom, cur_u_div, pos, size, nuchan);
        }
      }
    }
  }
  return 0;
}

static inline void tfluids_(Main_calcVelocityDivergenceCellBackward)(
    const real* u, const real* geom, real* grad_u, const real* grad_output,
    const int32_t* pos, const int32_t* size, const int32_t nuchan) {
  Int3 dims;
  dims.x = size[0];
  dims.y = size[1];
  dims.z = size[2];
  
  if (IsBlockedCell(geom, pos[0], pos[1], pos[2], dims)) {
    // geometry cells do not contribute any gradient (since UDiv(geometry) = 0).
    return;
  }
  
  // Now calculate the partial derivatives in each dimension.
  for (int32_t dim = 0; dim < nuchan; dim ++) {
    const int32_t uslice = dim * size[0] * size[1] * size[2];

    int32_t pos_p[3] = {pos[0], pos[1], pos[2]};
    pos_p[dim] += 1;   
    int32_t pos_n[3] = {pos[0], pos[1], pos[2]};
    pos_n[dim] -= 1;

    // Look at the neighbor to the right (pos) and to the left (neg).
    bool geomPos = false;
    bool geomNeg = false;
    if (pos[dim] == 0) {
      geomNeg = true;  // Treat going off the fluid as geometry.
    }
    if (pos[dim] == size[dim] - 1) {
      geomPos = true;  // Treat going off the fluid as geometry. 
    }
    if (pos[dim] > 0) {
      geomNeg = IsBlockedCell(geom, pos_n[0], pos_n[1], pos_n[2], dims);
    }
    if (pos[dim] < size[dim] - 1) { 
      geomPos = IsBlockedCell(geom, pos_p[0], pos_p[1], pos_p[2], dims);
    }

    if (geomPos and geomNeg) {
      continue;
    } else if (geomPos) {
#pragma omp atomic
      grad_u[uslice + IX(pos[0], pos[1], pos[2], dims)] +=
        grad_output[IX(pos[0], pos[1], pos[2], dims)];
#pragma omp atomic
      grad_u[uslice + IX(pos_n[0], pos_n[1], pos_n[2], dims)] -=
        grad_output[IX(pos[0], pos[1], pos[2], dims)];
    } else if (geomNeg) {
#pragma omp atomic
      grad_u[uslice + IX(pos_p[0], pos_p[1], pos_p[2], dims)] +=
        grad_output[IX(pos[0], pos[1], pos[2], dims)];
#pragma omp atomic
      grad_u[uslice + IX(pos[0], pos[1], pos[2], dims)] -=
        grad_output[IX(pos[0], pos[1], pos[2], dims)];
    } else {
#pragma omp atomic
      grad_u[uslice + IX(pos_p[0], pos_p[1], pos_p[2], dims)] +=
        grad_output[IX(pos[0], pos[1], pos[2], dims)] * static_cast<real>(0.5);
#pragma omp atomic
      grad_u[uslice + IX(pos_n[0], pos_n[1], pos_n[2], dims)] -=
        grad_output[IX(pos[0], pos[1], pos[2], dims)] * static_cast<real>(0.5);
    }
  }
}

static int tfluids_(Main_calcVelocityDivergenceBackward)(lua_State *L) {
  THTensor* grad_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* geom =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* grad_output = 
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (u->nDimension != 5 || grad_u->nDimension != 5 || geom->nDimension != 4 ||
      grad_output->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int32_t nbatch = u->size[0];
  const int32_t nuchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];

  // We will be accumulating gradient contributions into the grad_u tensor, so
  // we first need to zero it.
  THTensor_(zero)(grad_u);

  real* grad_u_data = THTensor_(data)(grad_u);
  const real* geom_data = THTensor_(data)(geom);
  const real* u_data = THTensor_(data)(u);
  const real* grad_output_data = THTensor_(data)(grad_output);

  int32_t b, z, y, x;
#pragma omp parallel for private(b, z, y, x) collapse(4)
  for (b = 0; b < nbatch; b++) {
    for (z = 0; z < zdim; z++) { 
      for (y = 0; y < ydim; y++) { 
        for (x = 0; x < xdim; x++) {
          const real* cur_u = &u_data[b * nuchan * zdim * ydim * xdim];
          const real* cur_geom = &geom_data[b * zdim * ydim * xdim];
          real* cur_grad_u = &grad_u_data[b * nuchan * zdim * ydim * xdim];
          const real* cur_grad_output =
              &grad_output_data[b * zdim * ydim * xdim];
          const int32_t pos[3] = {x, y, z};
          const int32_t size[3] = {xdim, ydim, zdim};

          tfluids_(Main_calcVelocityDivergenceCellBackward)(
              cur_u, cur_geom, cur_grad_u, cur_grad_output, pos, size, nuchan);
        }
      }
    }
  }
  return 0;
}

static int tfluids_(Main_solveLinearSystemPCG)(lua_State *L) {
  luaL_error(L, "ERROR: solveLinearSystemPCG not defined for CPU tensors.");
  return 0;
}

static const struct luaL_Reg tfluids_(Main__) [] = {
  {"advectScalar", tfluids_(Main_advectScalar)},
  {"advectVel", tfluids_(Main_advectVel)}, 
  {"vorticityConfinement", tfluids_(Main_vorticityConfinement)},
  {"averageBorderCells", tfluids_(Main_averageBorderCells)},
  {"setObstacleBcs", tfluids_(Main_setObstacleBcs)},
  {"interpField", tfluids_(Main_interpField)},
  {"drawVelocityField", tfluids_(Main_drawVelocityField)},
  {"loadTensorTexture", tfluids_(Main_loadTensorTexture)},
  {"calcVelocityUpdate", tfluids_(Main_calcVelocityUpdate)},
  {"calcVelocityUpdateBackward", tfluids_(Main_calcVelocityUpdateBackward)},
  {"calcVelocityDivergence", tfluids_(Main_calcVelocityDivergence)},
  {"calcVelocityDivergenceBackward",
   tfluids_(Main_calcVelocityDivergenceBackward)},
  {"solveLinearSystemPCG", tfluids_(Main_solveLinearSystemPCG)},
  {NULL, NULL}  // NOLINT
};

void tfluids_(Main_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, tfluids_(Main__), "tfluids");
}

#endif  // TH_GENERIC_FILE
