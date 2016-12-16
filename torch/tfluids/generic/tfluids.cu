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

#include <TH.h>
#include <THC.h>
#include <luaT.h>

#include <assert.h>
#include <cusparse.h>
#include <float.h>
#include <algorithm>
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#include "quadrants.h"  // typedef enum Quadrants

// This is REALLY ugly. But unfortunately cutorch_getstate() in
// cutorch/torch/util.h is not exposed externally. We could call
// cutorch.getState() from lua and pass in the struct into all the tfluids c
// functions (as Soumith did with nn and cunn), but I think this is also just
// as ugly. Instead lets just redefine cutorch_getstate and hope nothing
// breaks :-(

struct THCState* cutorch_getstate(lua_State* L) {
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "_state");
  struct THCState* state = reinterpret_cast<THCState*>(lua_touserdata(L, -1));
  lua_pop(L, 2);
  return state;
}

static cusparseHandle_t cusparse_handle = 0;

static void init_cusparse() {
  if (cusparse_handle == 0) {
    cusparseStatus_t status = cusparseCreate(&cusparse_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      THError("CUSPARSE Library initialization failed");
    }
  }
}

//******************************************************************************
// HELPER METHODS
//******************************************************************************

// We never want positions to go exactly to the border or exactly to the edge
// of an occupied piece of geometry. Therefore all rays will be truncated by
// a very small amount (hit_margin).
static const float hit_margin = 1e-5f;
static const float max_float = std::numeric_limits<float>::max();

// Get the integer index of the current voxel.
// Excerpt of comment from tfluids.cc:
// ... the (-0.5, 0, 0) position is the LEFT face of the first cell. Likewise
// (xdim - 0.5, ydim - 0.5, zdim - 0.5) is the upper bound of the grid (right
// at the corner). ...
__device__ __forceinline__ void GetPixelCenter(
    const float pos[3], int* ix, int* iy, int* iz) {
  *ix = (int)(pos[0] + 0.5f);
  *iy = (int)(pos[1] + 0.5f);
  *iz = (int)(pos[2] + 0.5f);
}

// geom is just used here as a reference to pass in domain size.
__device__ __forceinline__ bool IsOutOfDomain(
    const int i, const int j, const int k,
    const THCDeviceTensor<float, 3>& geom) {
  const int dimx = geom.getSize(2);
  const int dimy = geom.getSize(1);
  const int dimz = geom.getSize(0);
  return i < 0 || i >= dimx || j < 0 || j >= dimy || k < 0 || k >= dimz;
}

// geom is just used here as a reference to pass in domain size.
__device__ __forceinline__ bool IsOutOfDomainReal(
    const float pos[3], const THCDeviceTensor<float, 3>& geom) {
  const int dimx = geom.getSize(2);
  const int dimy = geom.getSize(1);
  const int dimz = geom.getSize(0);
  return (pos[0] <= -0.5f ||  // LHS of grid cell.
          pos[0] >= ((float)dimx - 0.5f) ||  // RHS of grid cell.
          pos[1] <= -0.5f ||
          pos[1] >= ((float)dimy - 0.5f) ||
          pos[2] <= -0.5f ||
          pos[2] >= ((float)dimz - 0.5f));
}

__device__ __forceinline__ bool IsBlockedCell(
    const THCDeviceTensor<float, 3>& geom, int i, int j, int k) {
  // Returns true if the cell is blocked.
  // Shouldn't be called on point outside the domain.
  assert(!tfluids_(IsOutOfDomain)(i, j, k, geom));
  return geom[k][j][i] == 1.0f;
}

__device__ __forceinline__ bool IsBlockedCellReal(
    const THCDeviceTensor<float, 3>& geom, const float pos[3]) {
  int ix, iy, iz;
  GetPixelCenter(pos, &ix, &iy, &iz);
  return IsBlockedCell(geom, ix, iy, iz);
}

__device__ __forceinline__ void ClampToDomain(
    const THCDeviceTensor<float, 3>& geom, int* ix, int* iy, int* iz) {
  const int dimx = geom.getSize(2);
  const int dimy = geom.getSize(1);
  const int dimz = geom.getSize(0);
  *ix = max(min(*ix, dimx - 1), 0);
  *iy = max(min(*iy, dimy - 1), 0);
  *iz = max(min(*iz, dimz - 1), 0);
}

__device__ __forceinline__ void ClampToDomainReal(
    float pos[3], const THCDeviceTensor<float, 3>& geom) {
  const float dimx = (float)geom.getSize(2);
  const float dimy = (float)geom.getSize(1);
  const float dimz = (float)geom.getSize(0);
  const float half = 0.5f;
  pos[0] = min(max(pos[0], -half + hit_margin), dimx - half - hit_margin);
  pos[1] = min(max(pos[1], -half + hit_margin), dimy - half - hit_margin);
  pos[2] = min(max(pos[2], -half + hit_margin), dimz - half - hit_margin);
}

__device__ __forceinline__ float length3(const float v[3]) {
  const float length_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (length_sq > 1e-6f) {
    return sqrt(length_sq);
  } else {
    return 0.0f;
  }
}

//******************************************************************************
// averageBorderCells
//******************************************************************************

// averageBorderCells CUDA kernel
__global__ void kernel_averageBorderCells(
    THCDeviceTensor<float, 4> in, THCDeviceTensor<float, 3> geom,
    THCDeviceTensor<float, 4> out, const int nchan, const int zdim,
    const int ydim, const int xdim, const bool two_dim) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int z = blockIdx.y;
  const int chan = blockIdx.z;
  if (pnt_id >= (in.getSize(2) * in.getSize(3))) {
    return;
  }
  const int y = pnt_id / in.getSize(3);
  const int x = pnt_id % in.getSize(3);
  const bool on_border = (!two_dim && (z == 0 || z == (zdim - 1))) ||
      y == 0 || y == (ydim - 1) ||
      x == 0 || x == (xdim - 1);

  if (!on_border) {
    // Copy the result and return.
    // NOTE: due to weird compile issues with the THCDeviceSubTensor function,
    // we have to set the in value to a temporary register before setting the
    // out value (i.e. this doesn't compile out[x] = in[x]).
    const float value = in[chan][z][y][x];
    out[chan][z][y][x] = value;
    return;
  }

  // We're a border pixel.
  // TODO(tompson): This is an O(n^3) iteration to find a small
  // sub-set of the pixels. Fix it. (same as C code).

  float sum = 0;
  int count = 0;
  for (int zoff = z - 1; zoff <= z + 1; zoff++) {
    for (int yoff = y - 1; yoff <= y + 1; yoff++) {
      for (int xoff = x - 1; xoff <= x + 1; xoff++) {
        if (zoff >= 0 && zoff < zdim && yoff >= 0 && yoff < ydim &&
            xoff >= 0 && xoff < xdim) {
          // The neighbor is on the image.
          if (geom[zoff][yoff][xoff] < 1e-6f) {
            // The neighbor is NOT geometry.
            count++;
            sum += in[chan][zoff][yoff][xoff];
          }
        }
      }
    }
  }
  if (count > 0) {
    out[chan][z][y][x] = sum / (float)(count);
  } else {
    // No non-geom pixels found. Just copy over result.
    const float value = in[chan][z][y][x];
    out[chan][z][y][x] = value;
  }
}

// averageBorderCells lua entry point.
static int tfluids_CudaMain_averageBorderCells(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  THCudaTensor* in = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* out = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  if (in->nDimension != 4 || geom->nDimension != 3) {
    luaL_error(L, "Input tensor should be 4D and geom should be 3D");
  }
  const int nchan = in->size[0];
  const int zdim = in->size[1];
  const int ydim = in->size[2];
  const int xdim = in->size[3];
  const bool two_dim = zdim == 1;

  THCDeviceTensor<float, 4> dev_in = toDeviceTensor<float, 4>(state, in);
  THCDeviceTensor<float, 3> dev_geom = toDeviceTensor<float, 3>(state, geom);
  THCDeviceTensor<float, 4> dev_out = toDeviceTensor<float, 4>(state, out);

  int nplane = dev_in.getSize(2) * dev_in.getSize(3);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_in.getSize(1), dev_in.getSize(0));
  dim3 block_size(nplane > 256 ? 256 : nplane);

  kernel_averageBorderCells<<<grid_size, block_size, 0,
                              THCState_getCurrentStream(state)>>>(
      dev_in, dev_geom, dev_out, nchan, zdim, ydim, xdim, two_dim);

  return 0;
}

//******************************************************************************
// setObstacleBCS
//******************************************************************************

// setObstacleBcs CUDA kernel
__global__ void kernel_setObstacleBcs(
    THCDeviceTensor<float, 4> U, THCDeviceTensor<float, 3> geom,
    const int zdim, const int ydim, const int xdim, const bool two_dim) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int z = blockIdx.y;
  const int chan = blockIdx.z;
  if (pnt_id >= (U.getSize(2) * U.getSize(3))) {
    return;
  }
  const int y = pnt_id / U.getSize(3);
  const int x = pnt_id % U.getSize(3);

  // For some reason (that completely baffles me) on a NVidia Titan X some of
  // device ids sometimes go higher than the limits! I have no idea why and
  // it only happens for this function.
  if (x >= xdim || y >= ydim || z >= zdim) {
    return;
  }
  if (two_dim && chan >= 2) {
    return;
  }
  if (!two_dim && chan >= 3) {
    return;
  }

  // Note: it should be one thread per channel so there shouldn't be any
  // collisions on U here.

  if (geom[z][y][x] == 1.0f) {
    // Zero out the velocity component.
    U[chan][z][y][x] = 0.0f;
  } else {
    // Don't update fluid cell velocities.
    return;
  }

  const int pos[3] = {x, y, z};
  const int size[3] = {xdim, ydim, zdim};
  int pos_p[3] = {pos[0], pos[1], pos[2]};
  pos_p[chan] += 1;
  int pos_n[3] = {pos[0], pos[1], pos[2]};
  pos_n[chan] -= 1;

  // Look positive direction.
  if (pos[chan] > 0 && geom[pos_n[2]][pos_n[1]][pos_n[0]] == 0.0f) {
    const float val = U[chan][pos_n[2]][pos_n[1]][pos_n[0]];
    U[chan][z][y][x] -= val;
  }
  // Look negative direction.
  if (pos[chan] < (size[chan] - 1) && 
      geom[pos_p[2]][pos_p[1]][pos_p[0]] == 0.0f) {
    const float val = U[chan][pos_p[2]][pos_p[1]][pos_p[0]];
    U[chan][z][y][x] -= val;
  }
}

// setObstacleBcs lua entry point.
static int tfluids_CudaMain_setObstacleBcs(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  THCudaTensor* U = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  if (U->nDimension != 4 || geom->nDimension != 3) {
    luaL_error(L, "Input tensor should be 4D and geom should be 3D");
  }
  const bool two_dim = U->size[0] == 2;
  const int zdim = U->size[1];
  const int ydim = U->size[2];
  const int xdim = U->size[3];
  if (two_dim && zdim != 1) {
    luaL_error(L, "2D velocity field but zdim != 1");
  }
  if (!two_dim && U->size[0] != 3) {
    luaL_error(L, "Number of velocity components should be 2 or 3.");
  }
  if (zdim != geom->size[0] || ydim != geom->size[1] || xdim != geom->size[2]) {
    luaL_error(L, "U / geom size mismatch");
  }

  THCDeviceTensor<float, 4> dev_U = toDeviceTensor<float, 4>(state, U);
  THCDeviceTensor<float, 3> dev_geom = toDeviceTensor<float, 3>(state, geom);

  int nplane = dev_U.getSize(2) * dev_U.getSize(3);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_U.getSize(1), dev_U.getSize(0));
  dim3 block_size(nplane > 256 ? 256 : nplane);

  kernel_setObstacleBcs<<<grid_size, block_size, 0,
                          THCState_getCurrentStream(state)>>>(
      dev_U, dev_geom, zdim, ydim, xdim, two_dim);
  return 0;
}

//******************************************************************************
// vorticityConfinement
//******************************************************************************

__device__ __forceinline__ void getCurl3D(
    const THCDeviceTensor<float, 4>& U, const int i,
    const int j, const int k, const int xdim,
    const int ydim, const int zdim, float curl[3]) {
  float dwdj, dudj;
  if (j == 0) {
    // Single sided diff (pos side).
    dwdj = U[2][k][j + 1][i] - U[2][k][j][i];
    dudj = U[0][k][j + 1][i] - U[0][k][j][i];
  } else if (j == ydim - 1) {
    // Single sided diff (neg side).
    dwdj = U[2][k][j][i] - U[2][k][j - 1][i];
    dudj = U[0][k][j][i] - U[0][k][j - 1][i];
  } else {
    // Central diff.
    dwdj = 0.5f * (U[2][k][j + 1][i] - U[2][k][j - 1][i]);
    dudj = 0.5f * (U[0][k][j + 1][i] - U[0][k][j - 1][i]);
  }

  float dwdi, dvdi;
  if (i == 0) {
    // Single sided diff (pos side).
    dwdi = U[2][k][j][i + 1] - U[2][k][j][i];
    dvdi = U[1][k][j][i + 1] - U[1][k][j][i];
  } else if (i == xdim - 1) {
    // Single sided diff (neg side).
    dwdi = U[2][k][j][i] - U[2][k][j][i - 1];
    dvdi = U[1][k][j][i] - U[1][k][j][i - 1];
  } else {
    // Central diff.
    dwdi = 0.5f * (U[2][k][j][i + 1] - U[2][k][j][i - 1]);
    dvdi = 0.5f * (U[1][k][j][i + 1] - U[1][k][j][i - 1]);
  }

  float dudk, dvdk;
  if (k == 0) {
    // Single sided diff (pos side).
    dudk = U[0][k + 1][j][i] - U[0][k][j][i];
    dvdk = U[1][k + 1][j][i] - U[1][k][j][i];
  } else if (k == zdim - 1) {
    // Single sided diff (neg side).
    dudk = U[0][k][j][i] - U[0][k - 1][j][i];
    dvdk = U[1][k][j][i] - U[1][k - 1][j][i];
  } else {
    // Central diff.
    dudk = 0.5f * (U[0][k + 1][j][i] - U[0][k - 1][j][i]);
    dvdk = 0.5f * (U[1][k + 1][j][i] - U[1][k - 1][j][i]);
  }

  curl[0] = dwdj - dvdk;
  curl[1] = dudk - dwdi;
  curl[2] = dvdi - dudj;
}

__device__ __forceinline__ float getCurl2D(
    const THCDeviceTensor<float, 4>& U, const int i,
    const int j, const int xdim, const int ydim) {
  float dvdi;
  const int k = 0;
  if (i == 0) {
    // Single sided diff (pos side).
    dvdi = U[1][k][j][i + 1] - U[1][k][j][i];
  } else if (i == xdim - 1) {
    // Single sided diff (neg side).
    dvdi = U[1][k][j][i] - U[1][k][j][i - 1];
  } else {
    // Central diff.
    dvdi = 0.5f * (U[1][k][j][i + 1] - U[1][k][j][i - 1]);
  }

  float dudj;
  if (j == 0) {
    // Single sided diff (pos side).
    dudj = U[0][k][j + 1][i] - U[0][k][j][i];
  } else if (j == ydim - 1) {
    // Single sided diff (neg side).
    dudj = U[0][k][j][i] - U[0][k][j - 1][i];
  } else {
    // Central diff.
    dudj = 0.5f * (U[0][k][j + 1][i] - U[0][k][j - 1][i]);
  }

  return dudj - dvdi;
}

// vorticityConfinementCurl CUDA kernel
__global__ void kernel_vorticityConfinementCurl(
    THCDeviceTensor<float, 4> U, THCDeviceTensor<float, 3> geom,
    const int zdim, const int ydim, const int xdim, const bool two_dim,
    THCDeviceTensor<float, 4> curl, THCDeviceTensor<float, 3> mag_curl) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = blockIdx.y;
  const int z = blockIdx.z;
  if (x >= U.getSize(3)) {
    return;
  }

  if (two_dim) {
    const float curl_val = getCurl2D(U, x, y, xdim, ydim);
    curl[0][z][y][x] = curl_val;
    mag_curl[z][y][x] = fabsf(curl_val);
  } else {
    float curl_val[3];
    getCurl3D(U, x, y, z, xdim, ydim, zdim, curl_val);
    curl[0][z][y][x] = curl_val[0];
    curl[1][z][y][x] = curl_val[1];
    curl[2][z][y][x] = curl_val[2];
    const float length_sq = curl_val[0] * curl_val[0] +
        curl_val[1] * curl_val[1] + curl_val[2] * curl_val[2];
    mag_curl[z][y][x] = (length_sq > FLT_EPSILON) ? sqrt(length_sq) : 0.0f;
  } 
}

// vorticityConfinement CUDA kernel
__global__ void kernel_vorticityConfinement(
    THCDeviceTensor<float, 4> U, THCDeviceTensor<float, 3> geom,
    const int zdim, const int ydim, const int xdim, const bool two_dim,
    THCDeviceTensor<float, 4> curl, THCDeviceTensor<float, 3> mag_curl,
    const float dt, const float scale) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = blockIdx.y;
  const int z = blockIdx.z;
  if (x >= U.getSize(3)) {
    return;
  }

  // Don't apply vorticity confinement to the borders.
  // TODO(tompson): We probably could if we tried hard enough and did single
  // sided FEM on the border for grad(||w||). But it's OK for now.
  if (x == 0 || x == xdim - 1 ||
      y == 0 || y == ydim - 1 ||
      (!two_dim && (z == 0 || z == zdim - 1))) {
    return;
  }

  // Don't perform any confinement in an obstacle.
  if (geom[z][y][x] == 1.0f) {
    return;
  }

  // Don't perform any confinement in a cell that borders an obstacle.
  // TODO(tompson): Not sure if this is correct. It's probably OK for now
  // only because the partial derivative for ||w|| is invalid for cells
  // that lie next to a geometry cell.
  // TODO(tompson): This is probably slow (all these accesses). It's probably OK
  // as a baseline implementation.
  if (geom[z][y][x - 1] == 1.0f || geom[z][y][x + 1] == 1.0f ||
      geom[z][y - 1][x] == 1.0f || geom[z][y + 1][x] == 1.0f ||
      (!two_dim && (geom[z - 1][y][x] == 1.0f || geom[z + 1][y][x] == 1.0f))) {
    return;
  }

  float forcex = 0.0f;
  float forcey = 0.0f;
  float forcez = 0.0f;
  if (two_dim) {
    // Find derivative of the magnitude of curl (n = grad |w|).
    // Where 'w' is the curl calculated above.
    float dwdx = 0.5f * (mag_curl[z][y][x + 1] - mag_curl[z][y][x - 1]);
    float dwdy = 0.5f * (mag_curl[z][y + 1][x] - mag_curl[z][y - 1][x]);
    const float length_sq = dwdx * dwdx + dwdy * dwdy;
    const float length = (length_sq > FLT_EPSILON) ? sqrtf(length_sq) : 0.0f;
    if (length > 1e-6f) {
      dwdx /= length;
      dwdy /= length;
    }
    const float v = curl[0][z][y][x];

    // N x w.
    forcex = dwdy * (-v);
    forcey = dwdx * v;
  } else {
    float dwdx = 0.5f * (mag_curl[z][y][x + 1] - mag_curl[z][y][x - 1]);
    float dwdy = 0.5f * (mag_curl[z][y + 1][x] - mag_curl[z][y - 1][x]);
    float dwdz = 0.5f * (mag_curl[z + 1][y][x] - mag_curl[z - 1][y][x]);

    const float length_sq = dwdx * dwdx + dwdy * dwdy + dwdz * dwdz;
    const float length = (length_sq > FLT_EPSILON) ? sqrtf(length_sq) : 0.0f;
    if (length > 1e-6f) {
      dwdx /= length;
      dwdy /= length;
      dwdz /= length;
    }

    const float Nx = dwdx;
    const float Ny = dwdy;
    const float Nz = dwdz;
   
    const float curlx = curl[0][z][y][x];
    const float curly = curl[1][z][y][x];
    const float curlz = curl[2][z][y][x]; 

    // N x w.
    forcex = Ny * curlz - Nz * curly;
    forcey = Nz * curlx - Nx * curlz;
    forcez = Nx * curly - Ny * curlx;
  }
  
  const float Ux = U[0][z][y][x];
  const float Uy = U[1][z][y][x];
  const float Uz = two_dim ? 0.0f : U[2][z][y][x];
  U[0][z][y][x] = Ux + forcex * scale * dt;
  U[1][z][y][x] = Uy + forcey * scale * dt;
  if (!two_dim) {
    U[2][z][y][x] = Uz + forcez * scale * dt;
  }
}

// vorticityConfinement lua entry point.
static int tfluids_CudaMain_vorticityConfinement(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  const float dt = (float)(lua_tonumber(L, 1));
  const float scale = (float)(lua_tonumber(L, 2));  
  THCudaTensor* U = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* curl = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  THCudaTensor* mag_curl = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));

  if (U->nDimension != 4 || geom->nDimension != 3 ||
      mag_curl->nDimension != 3) {
    luaL_error(L, "Incorrect input sizes.");
  }

  bool two_dim = U->size[0] == 2;
  const int xdim = static_cast<int>(geom->size[2]);
  const int ydim = static_cast<int>(geom->size[1]);
  const int zdim = static_cast<int>(geom->size[0]);

  if (two_dim && curl->nDimension != 3) {
     luaL_error(L, "Bad curl size.");
  }
  if (!two_dim && curl->nDimension != 4) {
     luaL_error(L, "Bad curl size.");
  }
  
  if (two_dim && zdim != 1) {
    luaL_error(L, "Incorrect input sizes.");
  }
  
  if (!two_dim && U->size[0] != 3) {
    luaL_error(L, "Incorrect input sizes.");
  }

  if (two_dim) {
    THCudaTensor_resize4d(state, curl, 1, zdim, ydim, xdim);
  }

  THCDeviceTensor<float, 4> dev_U = toDeviceTensor<float, 4>(state, U);
  THCDeviceTensor<float, 3> dev_geom = toDeviceTensor<float, 3>(state, geom);
  THCDeviceTensor<float, 4> dev_curl = toDeviceTensor<float, 4>(state, curl);
  THCDeviceTensor<float, 3> dev_mag_curl =
      toDeviceTensor<float, 3>(state, mag_curl);

  int nplane = dev_U.getSize(3);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_U.getSize(2), dev_U.getSize(1));
  dim3 block_size(nplane > 256 ? 256 : nplane);

  // As per the C++ code, calculate the curl in the first pass.
  kernel_vorticityConfinementCurl<<<grid_size, block_size, 0,
                                    THCState_getCurrentStream(state)>>>(
      dev_U, dev_geom, zdim, ydim, xdim, two_dim, dev_curl, dev_mag_curl);

  // Now apply vorticity confinement force in the second pass.
  kernel_vorticityConfinement<<<grid_size, block_size, 0,
                                THCState_getCurrentStream(state)>>>(
      dev_U, dev_geom, zdim, ydim, xdim, two_dim, dev_curl, dev_mag_curl, dt,
      scale);
  
  if (two_dim) {
    THCudaTensor_resize3d(state, curl, zdim, ydim, xdim);
  }

  return 0;
}

//******************************************************************************
// calcVelocityUpdate
//******************************************************************************

// calcVelocityUpdate CUDA kernel.
__global__ void kernel_calcVelocityUpdate(
    THCDeviceTensor<float, 5> delta_u, THCDeviceTensor<float, 4> p,
    THCDeviceTensor<float, 4> geom, const int nbatch, const int nuchan,
    const int zdim, const int ydim, const int xdim, const bool match_manta) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int dim = blockIdx.y;  // U-slice, i.e. Ux, Uy or Uz.
  const int batch = blockIdx.z;
  if (pnt_id >= (delta_u.getSize(2) * delta_u.getSize(3) *
      delta_u.getSize(4))) {
    return;
  }
  int x = pnt_id % delta_u.getSize(4);
  int y = (pnt_id / delta_u.getSize(4)) % delta_u.getSize(3);
  int z = pnt_id / (delta_u.getSize(3) * delta_u.getSize(4));

  const int pos[3] = {x, y, z};

  if (geom[batch][pos[2]][pos[1]][pos[0]] == 1.0f) {
    delta_u[batch][dim][pos[2]][pos[1]][pos[0]] = 0;
    return;
  }

  const int size[3] = {xdim, ydim, zdim};
  int pos_p[3] = {pos[0], pos[1], pos[2]};
  pos_p[dim] += 1;
  int pos_n[3] = {pos[0], pos[1], pos[2]};
  pos_n[dim] -= 1;

  // First annoying special case that happens on the border because of our
  // conversion to central velocities and because manta does not handle this
  // case properly.
  if (pos[dim] == 0 && match_manta) {
    if (geom[batch][pos_p[2]][pos_p[1]][pos_p[0]] == 1.0f &&
        geom[batch][pos[2]][pos[1]][pos[0]] == 0.0f) {
      delta_u[batch][dim][pos[2]][pos[1]][pos[0]] =
          p[batch][pos[2]][pos[1]][pos[0]] * 0.5f;
    } else {
      delta_u[batch][dim][pos[2]][pos[1]][pos[0]] =
          p[batch][pos_p[2]][pos_p[1]][pos_p[0]] * 0.5f;
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
    geomNeg = geom[batch][pos_n[2]][pos_n[1]][pos_n[0]] == 1.0f;
  }
  if (pos[dim] < size[dim] - 1) {
    geomPos = geom[batch][pos_p[2]][pos_p[1]][pos_p[0]] == 1.0f; 
  }

  // NOTE: The 0.5 below needs some explanation. We are exactly
  // mimicking CorrectVelocity() from
  // manta/source/pluging/pressure.cpp. In this function, all
  // updates are single sided, but they are done to the MAC cell
  // edges. When we convert to centered velocities, we therefore add
  // a * 0.5 term because we take the average.
  const float single_sided_gain = match_manta ? 0.5f : 1.0f;

  if (geomPos and geomNeg) {
    // There are 3 cases:
    // A) Cell is on the left border and has a right geom neighbor.
    // B) Cell is on the right border and has a left geom neighbor.
    // C) Cell has a right AND left geom neighbor.
    // In any of these cases the velocity should not receive a
    // pressure gradient (nowhere for the pressure to diffuse.
    delta_u[batch][dim][pos[2]][pos[1]][pos[0]] = 0;
  } else if (geomPos) {
    // There are 2 cases:
    // A) Cell is on the right border and there's fluid to the left.
    // B) Cell is internal but there is geom to the right.
    // In this case we need to do a single sided diff to the left.
    delta_u[batch][dim][pos[2]][pos[1]][pos[0]] = single_sided_gain *
        (p[batch][pos[2]][pos[1]][pos[0]] -
         p[batch][pos_n[2]][pos_n[1]][pos_n[0]]);
  } else if (geomNeg) {
    // There are 2 cases:
    // A) Cell is on the left border and there's fluid to the right.
    // B) Cell is internal but there is geom to the left.
    // In this case we need to do a single sided diff to the right.
    delta_u[batch][dim][pos[2]][pos[1]][pos[0]] = single_sided_gain *
        (p[batch][pos_p[2]][pos_p[1]][pos_p[0]] -
         p[batch][pos[2]][pos[1]][pos[0]]);
  } else {
    // The pixel is internal (not on border) with no geom neighbours.
    // Do a central diff.
    delta_u[batch][dim][pos[2]][pos[1]][pos[0]] = 0.5f *
        (p[batch][pos_p[2]][pos_p[1]][pos_p[0]] -
         p[batch][pos_n[2]][pos_n[1]][pos_n[0]]);
  }
}

// calcVelocityUpdate lua entry point.
static int tfluids_CudaMain_calcVelocityUpdate(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  THCudaTensor* delta_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool match_manta = static_cast<bool>(lua_toboolean(L, 4));

  if (delta_u->nDimension != 5 || p->nDimension != 4 || geom->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions. Expect delta_u: 5, p: 4, geom: 4.");
  }

  const int nbatch = delta_u->size[0];
  const int nuchan = delta_u->size[1];
  const int zdim = delta_u->size[2];
  const int ydim = delta_u->size[3];
  const int xdim = delta_u->size[4];

  THCDeviceTensor<float, 5> dev_delta_u =
      toDeviceTensor<float, 5>(state, delta_u);
  THCDeviceTensor<float, 4> dev_p = toDeviceTensor<float, 4>(state, p);
  THCDeviceTensor<float, 4> dev_geom = toDeviceTensor<float, 4>(state, geom);

  // One "thread" per output element.
  int nplane = dev_delta_u.getSize(2) * dev_delta_u.getSize(3) *
      dev_delta_u.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_delta_u.getSize(1),
                 dev_delta_u.getSize(0));
  dim3 block_size(nplane > 256 ? 256 : nplane);  
  
  kernel_calcVelocityUpdate<<<grid_size, block_size, 0,
                              THCState_getCurrentStream(state)>>>(
      dev_delta_u, dev_p, dev_geom, nbatch, nuchan, zdim, ydim, xdim,
      match_manta);
 
  return 0;
}

//******************************************************************************
// calcVelocityUpdateBackward
//******************************************************************************

// calcVelocityUpdateBackward CUDA kernel.
// TODO(tompson,kris): I'm sure these atomic add calls are slow! We should
// probably change this from a scatter to a gather op to avoid having to use
// them at all.
// (NVIDIA state that atomic operations on global memory are extremely slow) but
// on shared memory it is OK. So we could copy to shared first, use atomic ops
// there then use a small number of atomic ops back to global mem (probably
// rewriting it as a gather would be easier).
__global__ void kernel_calcVelocityUpdateBackward(
    THCDeviceTensor<float, 4> grad_p, THCDeviceTensor<float, 4> p,
    THCDeviceTensor<float, 4> geom, THCDeviceTensor<float, 5> grad_output,
    const int nbatch, const int nuchan, const int zdim, const int ydim,
    const int xdim, const bool match_manta) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int dim = blockIdx.y;  // U-slice, i.e. Ux, Uy or Uz.
  const int batch = blockIdx.z;
  if (pnt_id >= (grad_output.getSize(2) * grad_output.getSize(3) *
      grad_output.getSize(4))) {
    return;
  }
  int x = pnt_id % grad_output.getSize(4);
  int y = (pnt_id / grad_output.getSize(4)) % grad_output.getSize(3);
  int z = pnt_id / (grad_output.getSize(3) * grad_output.getSize(4));
  const int pos[3] = {x, y, z};
  if (geom[batch][pos[2]][pos[1]][pos[0]] == 1.0f) {
    // No gradient contribution from blocked cells (since U(blocked) == 0).
    return;
  }
  const int size[3] = {xdim, ydim, zdim};
  int pos_p[3] = {pos[0], pos[1], pos[2]};
  pos_p[dim] += 1;
  int pos_n[3] = {pos[0], pos[1], pos[2]};
  pos_n[dim] -= 1;

  if (pos[dim] == 0 && match_manta) {
    if (geom[batch][pos_p[2]][pos_p[1]][pos_p[0]] == 1.0f &&
        geom[batch][pos[2]][pos[1]][pos[0]] == 0.0f) {
      atomicAdd(&grad_p[batch][pos[2]][pos[1]][pos[0]], 0.5f *
          grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
    } else {
      atomicAdd(&grad_p[batch][pos_p[2]][pos_p[1]][pos_p[0]], 0.5f *
          grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
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
    geomNeg = geom[batch][pos_n[2]][pos_n[1]][pos_n[0]] == 1.0f;
  }
  if (pos[dim] < size[dim] - 1) {
    geomPos = geom[batch][pos_p[2]][pos_p[1]][pos_p[0]] == 1.0f; 
  }
  
  const float single_sided_gain = match_manta ? 0.5f : 1.0f;
  if (geomPos and geomNeg) {
    // Output velocity update is zero.
    // --> No gradient contribution from this case (since delta_u == 0).
  } else if (geomPos) {
    // Single sided diff to the left --> Spread the gradient contribution.
    atomicAdd(&grad_p[batch][pos[2]][pos[1]][pos[0]], single_sided_gain *
        grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
    atomicAdd(&grad_p[batch][pos_n[2]][pos_n[1]][pos_n[0]], -single_sided_gain *
        grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
  } else if (geomNeg) {
    // Single sided diff to the right --> Spread the gradient contribution.
    atomicAdd(&grad_p[batch][pos_p[2]][pos_p[1]][pos_p[0]], single_sided_gain *
        grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
    atomicAdd(&grad_p[batch][pos[2]][pos[1]][pos[0]], -single_sided_gain *
        grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
  } else {
    // Central diff --> Spread the gradient contribution.
    atomicAdd(&grad_p[batch][pos_p[2]][pos_p[1]][pos_p[0]], 0.5f *
        grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
    atomicAdd(&grad_p[batch][pos_n[2]][pos_n[1]][pos_n[0]], -0.5f *
        grad_output[batch][dim][pos[2]][pos[1]][pos[0]]);
  }
}

// calcVelocityUpdate lua entry point.
static int tfluids_CudaMain_calcVelocityUpdateBackward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  THCudaTensor* grad_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  const bool match_manta = static_cast<bool>(lua_toboolean(L, 5));

  if (grad_output->nDimension != 5 || p->nDimension != 4 ||
      geom->nDimension != 4 || grad_p->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }

  // We will be accumulating gradient contributions into the grad_p tensor, so
  // we first need to zero it.
  THCudaTensor_zero(state, grad_p);

  const int nbatch = grad_p->size[0];
  const int zdim = grad_p->size[1];
  const int ydim = grad_p->size[2];
  const int xdim = grad_p->size[3];
  const int nuchan = grad_output->size[1];

  THCDeviceTensor<float, 5> dev_grad_output =
      toDeviceTensor<float, 5>(state, grad_output);
  THCDeviceTensor<float, 4> dev_grad_p =
      toDeviceTensor<float, 4>(state, grad_p);
  THCDeviceTensor<float, 4> dev_p = toDeviceTensor<float, 4>(state, p);
  THCDeviceTensor<float, 4> dev_geom = toDeviceTensor<float, 4>(state, geom);

  // One "thread" per output element.
  int nplane = dev_grad_output.getSize(2) * dev_grad_output.getSize(3) *
      dev_grad_output.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_grad_output.getSize(1),
                 dev_grad_output.getSize(0));
  dim3 block_size(nplane > 256 ? 256 : nplane);

  kernel_calcVelocityUpdateBackward<<<grid_size, block_size, 0,
                                      THCState_getCurrentStream(state)>>>(
      dev_grad_p, dev_p, dev_geom, dev_grad_output, nbatch, nuchan, zdim,
      ydim, xdim, match_manta); 

  return 0;
}

//******************************************************************************
// calcVelocityDivergence
//******************************************************************************

// calcVelocityDivergence CUDA kernel.
__global__ void kernel_calcVelocityDivergence(
    THCDeviceTensor<float, 5> u, THCDeviceTensor<float, 4> geom,
    THCDeviceTensor<float, 4> u_div, const int nbatch, const int nuchan,
    const int zdim, const int ydim, const int xdim) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int z = blockIdx.y;
  const int batch = blockIdx.z;
  if (pnt_id >= (u.getSize(3) * u.getSize(4))) {
    return;
  }
  int x = pnt_id % u.getSize(4);
  int y = pnt_id / u.getSize(4);
  const int pos[3] = {x, y, z};

  // Zero the output divergence.
  u_div[batch][pos[2]][pos[1]][pos[0]] = 0.0f;

  if (geom[batch][pos[2]][pos[1]][pos[0]] == 1.0f) {
    // Divergence INSIDE geometry is zero always, or we don't try to minimize it
    // during training.
    return;
  }

  const int size[3] = {xdim, ydim, zdim};

  // Now calculate the partial derivatives in each dimension.
  for (int dim = 0; dim < nuchan; dim ++) {
    int pos_p[3] = {pos[0], pos[1], pos[2]};
    pos_p[dim] += 1;
    int pos_n[3] = {pos[0], pos[1], pos[2]};
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
      geomNeg = geom[batch][pos_n[2]][pos_n[1]][pos_n[0]] == 1.0f;
    }
    if (pos[dim] < size[dim] - 1) {
      geomPos = geom[batch][pos_p[2]][pos_p[1]][pos_p[0]] == 1.0f;
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
      u_div[batch][pos[2]][pos[1]][pos[0]] +=
          (u[batch][dim][pos[2]][pos[1]][pos[0]] -
           u[batch][dim][pos_n[2]][pos_n[1]][pos_n[0]]);
    } else if (geomNeg) {
      // There are 2 cases:
      // A) Cell is on the left border and there's fluid to the right.
      // B) Cell is internal but there is geom to the left.
      // In this case we need to do a single sided diff to the right.
      u_div[batch][pos[2]][pos[1]][pos[0]] +=
          (u[batch][dim][pos_p[2]][pos_p[1]][pos_p[0]] -
           u[batch][dim][pos[2]][pos[1]][pos[0]]);
    } else {
      // The pixel is internal (not on border) with no geom neighbours.
      // Do a central diff.
      u_div[batch][pos[2]][pos[1]][pos[0]] += 0.5f *
          (u[batch][dim][pos_p[2]][pos_p[1]][pos_p[0]] -
           u[batch][dim][pos_n[2]][pos_n[1]][pos_n[0]]);
    }
  }
}

// calcVelocityDivergence lua entry point.
static int tfluids_CudaMain_calcVelocityDivergence(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  THCudaTensor* u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* u_div = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (u->nDimension != 5 || u_div->nDimension != 4 || geom->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int nbatch = u->size[0];
  const int nuchan = u->size[1];
  const int zdim = u->size[2];
  const int ydim = u->size[3];
  const int xdim = u->size[4];

  THCDeviceTensor<float, 5> dev_u = toDeviceTensor<float, 5>(state, u);
  THCDeviceTensor<float, 4> dev_u_div = toDeviceTensor<float, 4>(state, u_div);
  THCDeviceTensor<float, 4> dev_geom = toDeviceTensor<float, 4>(state, geom);

  // One "thread" per output element.
  int nplane = dev_u.getSize(3) * dev_u.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_u.getSize(2), dev_u.getSize(0));
  dim3 block_size(nplane > 256 ? 256 : nplane);

  kernel_calcVelocityDivergence<<<grid_size, block_size, 0,
                                  THCState_getCurrentStream(state)>>>(
      dev_u, dev_geom, dev_u_div, nbatch, nuchan, zdim,
      ydim, xdim);

  return 0;
}

//******************************************************************************
// calcVelocityDivergenceBackward
//******************************************************************************

// calcVelocityDivergenceBackward CUDA kernel.
__global__ void kernel_calcVelocityDivergenceBackward(
    THCDeviceTensor<float, 5> u, THCDeviceTensor<float, 4> geom,
    THCDeviceTensor<float, 5> grad_u, THCDeviceTensor<float, 4> grad_output,
    const int nbatch, const int nuchan, const int zdim, const int ydim,
    const int xdim) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int z = blockIdx.y;
  const int batch = blockIdx.z;
  if (pnt_id >= (u.getSize(3) * u.getSize(4))) {
    return;
  }
  int x = pnt_id % u.getSize(4);
  int y = pnt_id / u.getSize(4);
  const int pos[3] = {x, y, z};

  if (geom[batch][pos[2]][pos[1]][pos[0]] == 1.0f) {
    // geometry cells do not contribute any gradient (since UDiv(geometry) = 0).
    return;
  }

  const int size[3] = {xdim, ydim, zdim};

  // Now calculate the partial derivatives in each dimension.
  for (int dim = 0; dim < nuchan; dim ++) {
    int pos_p[3] = {pos[0], pos[1], pos[2]};
    pos_p[dim] += 1;
    int pos_n[3] = {pos[0], pos[1], pos[2]};
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
      geomNeg = geom[batch][pos_n[2]][pos_n[1]][pos_n[0]] == 1.0f;
    }
    if (pos[dim] < size[dim] - 1) {
      geomPos = geom[batch][pos_p[2]][pos_p[1]][pos_p[0]] == 1.0f;
    }

    if (geomPos and geomNeg) {
      continue;
    } else if (geomPos) {
      atomicAdd(&grad_u[batch][dim][pos[2]][pos[1]][pos[0]],
        grad_output[batch][pos[2]][pos[1]][pos[0]]);
      atomicAdd(&grad_u[batch][dim][pos_n[2]][pos_n[1]][pos_n[0]],
        -grad_output[batch][pos[2]][pos[1]][pos[0]]);
    } else if (geomNeg) {
      atomicAdd(&grad_u[batch][dim][pos_p[2]][pos_p[1]][pos_p[0]],
        grad_output[batch][pos[2]][pos[1]][pos[0]]);
      atomicAdd(&grad_u[batch][dim][pos[2]][pos[1]][pos[0]],
        -grad_output[batch][pos[2]][pos[1]][pos[0]]);
    } else {
      atomicAdd(&grad_u[batch][dim][pos_p[2]][pos_p[1]][pos_p[0]],
        grad_output[batch][pos[2]][pos[1]][pos[0]] * 0.5f);
      atomicAdd(&grad_u[batch][dim][pos_n[2]][pos_n[1]][pos_n[0]],
        grad_output[batch][pos[2]][pos[1]][pos[0]] * -0.5f);
    }
  }
}


// calcVelocityDivergenceBackward lua entry point.
static int tfluids_CudaMain_calcVelocityDivergenceBackward(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  THCudaTensor* grad_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));

  // Just do a basic dim assert, everything else goes in the lua code.
  if (u->nDimension != 5 || grad_u->nDimension != 5 || geom->nDimension != 4 ||
      grad_output->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const int nbatch = u->size[0];
  const int nuchan = u->size[1];
  const int zdim = u->size[2];
  const int ydim = u->size[3];
  const int xdim = u->size[4];

  // We will be accumulating gradient contributions into the grad_u tensor, so
  // we first need to zero it.
  THCudaTensor_zero(state, grad_u);

  THCDeviceTensor<float, 5> dev_u = toDeviceTensor<float, 5>(state, u);
  THCDeviceTensor<float, 5> dev_grad_u =
      toDeviceTensor<float, 5>(state, grad_u);
  THCDeviceTensor<float, 4> dev_geom = toDeviceTensor<float, 4>(state, geom);
  THCDeviceTensor<float, 4> dev_grad_output =
      toDeviceTensor<float, 4>(state, grad_output);

  // One "thread" per output element.
  int nplane = dev_u.getSize(3) * dev_u.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_u.getSize(2), dev_u.getSize(0));
  dim3 block_size(nplane > 256 ? 256 : nplane);

  kernel_calcVelocityDivergenceBackward<<<grid_size, block_size, 0,
                                          THCState_getCurrentStream(state)>>>(
      dev_u, dev_geom, dev_grad_u, dev_grad_output, nbatch, nuchan, zdim,
      ydim, xdim);

  return 0;
}

//******************************************************************************
// solveLinearSystemPCG
//******************************************************************************

// solveLinearSystemPCG lua entry point.
static int tfluids_CudaMain_solveLinearSystemPCG(lua_State *L) {
  init_cusparse();  // No op if already initialized.
/*
  THCState* state = cutorch_getstate(L);
  THCudaTensor* delta_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));

  if (delta_u->nDimension != 5 || p->nDimension != 4 || geom->nDimension != 4 ||
      u->nDimension != 5) {
    luaL_error(L, "Incorrect dimensions. "
                  "Expect delta_u: 5, p: 4, geom: 4, u: 5.");
  }
  const int nbatch = delta_u->size[0];
  const int nuchan = delta_u->size[1];  // Can only be 2 or 3.
  const int zdim = delta_u->size[2];
  const int ydim = delta_u->size[3];
  const int xdim = delta_u->size[4];
  THCDeviceTensor<float, 5> dev_delta_u =
      toDeviceTensor<float, 5>(state, delta_u);
  THCDeviceTensor<float, 4> dev_p = toDeviceTensor<float, 4>(state, p);
  THCDeviceTensor<float, 4> dev_geom = toDeviceTensor<float, 4>(state, geom);
  THCDeviceTensor<float, 5> dev_u = toDeviceTensor<float, 5>(state, u);

  if (nuchan != 2 && nuchan != 3) {
    luaL_error(L, "Incorrect number of velocity channels.");
  }

  // Get raw ptrs.
  const float* ptr_delta_u = dev_delta_u.data();
  const float* ptr_p = dev_p.data();
  const float* ptr_geom = dev_geom.data();
  const float* ptr_u = dev_u.data();

  // MAKE SURE THE cuSPARSE LIBRARY IS BOUND CORRECTLY. DONOTSUBMIT.
  cusparseMatDescr_t descr = 0;  // DONOTSUBMIT
  cusparseCreateMatDescr(&descr);  // DONOTSUBMIT
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);  // DONOTSUBMIT
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);  // DONOTSUBMIT
*/
  luaL_error(L, "ERROR: solveLinearSystemPCG not implemented");  // DONOTSUBMIT
 
  return 0;
}

//******************************************************************************
// HELPER METHODS FOR ADVECTION ROUTINES
//******************************************************************************

// I HATE doing this, but I copied this code from here:
// https://github.com/erich666/GraphicsGems/blob/master/gems/RayBox.c
// And modified it (there were actually a few numerical precision bugs).
// I tested the hell out of it, so it seems to work.
//
// @param hit_margin - value >= 0 describing margin added to hit to
// prevent interpenetration.
__device__ __forceinline__ bool HitBoundingBoxCUDA(
    const float minB[3], const float maxB[3],  // box
    const float origin[3], const float dir[3],  // ray
    float coord[3]) {  // hit point.
  char inside = true;
  Quadrants quadrant[3];
  int i;
  int whichPlane;
  float maxT[3];
  float candidate_plane[3];

  // Find candidate planes; this loop can be avoided if rays cast all from the
  // eye (assume perpsective view).
  for (i = 0; i < 3; i++) {
    if (origin[i] < minB[i]) {
      quadrant[i] = LEFT;
      candidate_plane[i] = minB[i];
      inside = false;
    } else if (origin[i] > maxB[i]) {
      quadrant[i] = RIGHT;
      candidate_plane[i] = maxB[i];
      inside = false;
    } else {
      quadrant[i] = MIDDLE;
    }
  }

  // Ray origin inside bounding box.
  if (inside) {
    for (i = 0; i < 3; i++) {
      coord[i] = origin[i];
    }
    return true;
  }

  // Calculate T distances to candidate planes.
  for (i = 0; i < 3; i++) {
    if (quadrant[i] != MIDDLE && dir[i] != 0.0f) { 
      maxT[i] = (candidate_plane[i] - origin[i]) / dir[i];
    } else {
      maxT[i] = -1.0f;
    }
  }

  // Get largest of the maxT's for final choice of intersection.
  whichPlane = 0;
  for (i = 1; i < 3; i++) {
    if (maxT[whichPlane] < maxT[i]) {
      whichPlane = i;
    }
  }

  // Check final candidate actually inside box and calculate the coords (if
  // not).
  if (maxT[whichPlane] < 0.0f) {
    return false;
  }

  const float err_tol = 1e-6f;
  for (i = 0; i < 3; i++) {
    if (whichPlane != i) {
      coord[i] = origin[i] + maxT[whichPlane] * dir[i];
      if (coord[i] < (minB[i] - err_tol) || coord[i] > (maxB[i] + err_tol)) {
        return false;
      }
    } else {
      coord[i] = candidate_plane[i];
    }
  }

  return true;
}   

// calcRayBoxIntersection will calculate the intersection point for the ray
// starting at pos, and pointing along dt (which should be unit length).
// The box is size 1 and is centered at ctr.
//
// This is really just a wrapper around the function above.
__device__ __forceinline__ bool calcRayBoxIntersectionCUDA(
    const float pos[3], const float dt[3],
    const float ctr[3], const float hit_margin, float ipos[3]) {
  assert(hit_margin >= 0);
  float box_min[3];
  box_min[0] = ctr[0] - 0.5f - hit_margin;
  box_min[1] = ctr[1] - 0.5f - hit_margin;
  box_min[2] = ctr[2] - 0.5f - hit_margin;
  float box_max[3];
  box_max[0] = ctr[0] + 0.5f + hit_margin;
  box_max[1] = ctr[1] + 0.5f + hit_margin;
  box_max[2] = ctr[2] + 0.5f + hit_margin;

  return HitBoundingBoxCUDA(box_min, box_max,  // box
                            pos, dt,  // ray
                            ipos);
}

// calcRayBorderIntersection will calculate the intersection point for the ray
// starting at pos and pointing to next_pos.
//
// IMPORTANT: This function ASSUMES that the ray actually intersects. Nasty
// things will happen if it does not.
// EDIT(tompson, 09/25/16): This is so important that we'll actually double
// check the input coords anyway.
__device__ __forceinline__ bool calcRayBorderIntersectionCUDA(
    const float pos[3], const float next_pos[3],
    const THCDeviceTensor<float, 3>& geom, const float hit_margin,
    float ipos[3]) {
  assert(hit_margin >= 0);

  // The source location should be INSIDE the boundary.
  assert(!IsOutOfDomainReal(pos, geom));
  
  // The target location should be OUTSIDE the boundary.
  assert(IsOutOfDomainReal(next_pos, geom));

  const float dimsx = (float)geom.getSize(2);
  const float dimsy = (float)geom.getSize(1);
  const float dimsz = (float)geom.getSize(0);

  // Calculate the minimum step length to exit each face and then step that
  // far. The line equation is:
  //   P = gamma * (next_pos - pos) + pos.
  // So calculate gamma required to make P < -0.5 + margin for each dim
  // independently.
  //   P_i = -0.5+m --> -0.5+m - pos_i = gamma * (next_pos_i - pos_i)
  //              --> gamma_i = (-0.5+m - pos_i) / (next_pos_i - pos_i)
  float min_step = max_float;
  if (next_pos[0] <= -0.5f) {  // left face of cell.
    const float dx = next_pos[0] - pos[0];
    if (dx > 1e-6f || dx < -1e-6f) {
      const float xstep = (-0.5f + hit_margin - pos[0]) / dx;
      min_step = min(min_step, xstep);
    }
  }
  if (next_pos[1] <= -0.5f) {
    const float dy = next_pos[1] - pos[1];
    if (dy > 1e-6f || dy < -1e-6f) {
      const float ystep = (-0.5f + hit_margin - pos[1]) / dy;
      min_step = min(min_step, ystep);
    }
  }
  if (next_pos[2] <= -0.5f) {
    const float dz = next_pos[2] - pos[2];
    if (dz > 1e-6f || dz < -1e-6f) {
      const float zstep = (-0.5f + hit_margin - pos[2]) / dz;
      min_step = min(min_step, zstep);
    }
  }
  // Also calculate the min step to exit a positive face.
  //   P_i = dim - 0.5 - m --> dim - 0.5 - m - pos_i =
  //                             gamma * (next_pos_i - pos_i)
  //                       --> gamma = (dim - 0.5 - m - pos_i) /
  //                                   (next_pos_i - pos_i)
  if (next_pos[0] >= dimsx - 0.5f) {  // right face of cell.
    const float dx = next_pos[0] - pos[0];
    if (dx > 1e-6f || dx < -1e-6f) {
      const float xstep = (dimsx - 0.5f - hit_margin - pos[0]) / dx;
      min_step = min(min_step, xstep);
    }
  }
  if (next_pos[1] >= dimsy - 0.5f) {
    const float dy = next_pos[1] - pos[1];
    if (dy > 1e-6f || dy < -1e-6f) {
      const float ystep = (dimsy - 0.5f - hit_margin - pos[1]) / dy;
      min_step = min(min_step, ystep);
    }
  }
  if (next_pos[2] >= dimsz - 0.5f) {
    const float dz = next_pos[2] - pos[2];
    if (dz > 1e-6f || dz < -1e-6f) {
      const float zstep = (dimsz - 0.5f - hit_margin - pos[2]) / dz;
      min_step = min(min_step, zstep);
    }
  }
  if (min_step < 0 || min_step >= max_float) {
    return false;
  }

  // Take the minimum step.
  ipos[0] = min_step * (next_pos[0] - pos[0]) + pos[0];
  ipos[1] = min_step * (next_pos[1] - pos[1]) + pos[1];
  ipos[2] = min_step * (next_pos[2] - pos[2]) + pos[2];

  return true;
}

// The following function performs a line trace along the displacement vector
// and returns either:
// a) The position 'p + delta' if NO geometry is found on the line trace. or
// b) The position at the first geometry blocker along the path.
// The search is exhaustive (i.e. O(n) in the length of the displacement vector)
//
// Note: the returned position is NEVER in geometry or outside the bounds. We
// go to great lengths to ensure this.
//
// TODO(tompsion): This is probably not efficient at all.
// It also has the potential to miss geometry along the path if the width
// of the geometry is less than 1 grid.
//
// NOTE: Pos = (0, 0, 0) is the CENTER of the first grid cell.
//       Pos = (0.5, 0, 0) is the x face between the first 2 cells.
__device__ __forceinline__ bool calcLineTraceCUDA(
    const float pos[3], const float delta[3],
    const THCDeviceTensor<float, 3>& geom, float new_pos[3]) {
  // If we're ALREADY in a geometry segment (or outside the domain) then a lot
  // of logic below with fail. This function should only be called on fluid
  // cells!
  assert(!IsOutOfDomainReal(pos, geom) && !IsBlockedCellReal(geom, pos));

  new_pos[0] = pos[0];
  new_pos[1] = pos[1];
  new_pos[2] = pos[2];

  const float length = length3(delta);
  if (length <= 1e-6f) {
    // We're not being asked to step anywhere. Return false and copy the pos.
    // (copy already done above).
    return false;
  }
  // Figure out the step size in x, y and z for our marching.
  float dt[3];
  dt[0] = delta[0] / length;  // Recall: we've already check for div zero.
  dt[1] = delta[1] / length;
  dt[2] = delta[2] / length;

  // Otherwise, we start the line search, by stepping a unit length along the
  // vector and checking the neighbours.
  //
  // A few words about the implementation (because it's complicated and perhaps
  // needlessly so). We maintain a VERY important loop invariant: new_pos is
  // NEVER allowed to enter solid geometry or go off the domain. next_pos
  // is the next step's tentative location, and we will always try and back
  // it off to the closest non-geometry valid cell before updating new_pos.
  //
  // We will also go to great lengths to ensure this loop invariant is
  // correct (probably at the expense of speed).
  float cur_length = 0;
  float next_pos[3];  // Tentative step location.

  // TODO(tompson): This while loop is likely REALLY slow and stupid in CUDA.
  // This is just a straight C++ to CUDA port, but likely we need to do
  // something smarter (because all threads are going to be in lock-step
  // throughout the entire loop and we have lots and lots of branching
  // conditionals).

  while (cur_length < (length - hit_margin)) {
    // We haven't stepped far enough. So take a step.
    float cur_step = min(length - cur_length, 1.0f);
    next_pos[0] = new_pos[0] + cur_step * dt[0];
    next_pos[1] = new_pos[1] + cur_step * dt[1];
    next_pos[2] = new_pos[2] + cur_step * dt[2];

    // Check to see if we went too far.
    // TODO(tompson): This is not correct, we might skip over small
    // pieces of geometry if the ray brushes against the corner of a
    // occupied voxel, but doesn't land in it. Fix this (it's very rare though).
  
    // There are two possible cases. We've either stepped out of the domain
    // or entered a blocked cell.
    if (IsOutOfDomainReal(next_pos, geom)) {
      // Case 1. 'next_pos' exits the grid.
      float ipos[3];
      const bool hit = calcRayBorderIntersectionCUDA(new_pos, next_pos, geom, 
                                                     hit_margin, ipos);
      if (!hit) {
        // This is an EXTREMELY rare case. It happens once or twice during
        // training. It happens because either the ray is almost parallel
        // to the domain boundary, OR floating point round-off causes the
        // intersection test to fail.
        // In this case, fall back to simply clamping next_pos inside the domain
        // boundary. It's not ideal, but better than a hard failure.
        ipos[0] = next_pos[0];
        ipos[1] = next_pos[1];
        ipos[2] = next_pos[2];
        ClampToDomainReal(ipos, geom);
      }

      // Do some sanity checks. I'd rather be slow and correct...
      // The logic above should aways put ipos back inside the simulation
      // domain.
      assert(!IsOutOfDomainReal(ipos, geom));

      if (!IsBlockedCellReal(geom, ipos)) {
        // OK to return here (i.e. we're up against the border and not
        // in a blocked cell).
        new_pos[0] = ipos[0];
        new_pos[1] = ipos[1];
        new_pos[2] = ipos[2];
        return true;
      } else {
        // Otherwise, we hit the border boundary, but we entered a blocked cell.
        // Continue on to case 2.
        next_pos[0] = ipos[0];
        next_pos[1] = ipos[1];
        next_pos[2] = ipos[2];
      }
    }
    if (IsBlockedCellReal(geom, next_pos)) {
      // Case 2. next_pos enters a blocked cell.

      // If the source of the ray starts in a blocked cell, we'll never exit
      // the while loop below, also our loop invariant is that new_pos is
      // NEVER allowed to enter a geometry cell. So failing this test means
      // our logic is broken.
      assert(!IsBlockedCellReal(geom, new_pos));
      int count = 0;
      const int max_count = 100;
      static_cast<void>(max_count);
      while (IsBlockedCellReal(geom, next_pos)) {
        // Calculate the center of the blocker cell.
        float next_pos_ctr[3];
        int ix, iy, iz;
        GetPixelCenter(next_pos, &ix, &iy, &iz);
        next_pos_ctr[0] = (float)(ix);
        next_pos_ctr[1] = (float)(iy);
        next_pos_ctr[2] = (float)(iz);
        
        // Sanity check. This is redundant because IsBlockedCellReal USES
        // GetPixelCenter to sample the geometry field. But keep this here
        // just in case the implementation changes.
        assert(IsBlockedCellReal(geom, next_pos_ctr));

        // Center of blocker cell should not be out of the domain.
        assert(!IsOutOfDomainReal(next_pos_ctr, geom));

        float ipos[3];
        const bool hit = calcRayBoxIntersectionCUDA(new_pos, dt, next_pos_ctr,
                                                    hit_margin, ipos);
        // Hard assert if we didn't hit (even on release builds) because we
        // should have hit the aabbox!
        if (!hit) {
          // EDIT: This can happen in very rare cases if the ray box
          // intersection test fails because of floating point round off.
          // It can also happen if the simulation becomes unstable (maybe with a
          // poorly trained model) and the velocity values are extremely high.
          
          // In this case, fall back to simply returning new_pos (for which the
          // loop invariant guarantees is a valid point).
          next_pos[0] = new_pos[0];
          next_pos[1] = new_pos[1];
          next_pos[2] = new_pos[2];
          return true;
        }
  
        next_pos[0] = ipos[0];
        next_pos[1] = ipos[1];
        next_pos[2] = ipos[2];

        // There's a nasty corner case here. It's when the cell we were trying
        // to step to WAS a blocker, but the ray passed through a blocker to get
        // there (i.e. our step size didn't catch the first blocker). If this is
        // the case we need to do another intersection test, but this time with
        // the ray point destination that is the closer cell.
        // --> There's nothing to do. The outer while loop will try another
        // intersection for us.

        // A basic test to make sure we never spin here indefinitely (I've
        // never seen it, but just in case).
        count++;
        assert(count < max_count);
      }
      
      // At this point next_pos is guaranteed to be within the domain and
      // not within a solid cell. 
      new_pos[0] = next_pos[0];
      new_pos[1] = next_pos[1];
      new_pos[2] = next_pos[2];

      // Do some sanity checks.
      assert(!IsBlockedCellReal(geom, new_pos));
      assert(!IsOutOfDomainReal(new_pos, geom));
      return true;
    }

    // Otherwise, update the position to the current step location.
    new_pos[0] = next_pos[0];
    new_pos[1] = next_pos[1];
    new_pos[2] = next_pos[2];

    // Do some sanity checks and check the loop invariant.
    assert(!IsBlockedCellReal(geom, new_pos));
    assert(!IsOutOfDomainReal(new_pos, geom));

    cur_length += cur_step;
  }

  // Finally, yet another set of checks, just in case.
  assert(!IsOutOfDomainReal(new_pos, geom));
  assert(!IsBlockedCellReal(geom, new_pos));

  return false;
}

__device__ __forceinline__ void MixWithGeomCUDA(
    const float a, const float b, const bool a_geom, const bool b_geom,
    const bool sample_into_geom, const float t, float* interp_val,
    bool* interp_geom) {
  if (sample_into_geom || (!a_geom && !b_geom)) {
    *interp_geom = false;
    *interp_val = (1.0f - t) * a + t * b;
  } else if (a_geom && !b_geom) {
    *interp_geom = false;
    *interp_val = b;  // a is geometry, return b.
  } else if (b_geom && !a_geom) {
    *interp_geom = false;
    *interp_val = a;  // b is geometry, return a.
  } else {
    *interp_geom = true;  // both a and b are geom.
    *interp_val = 0.0f;
  }
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
__device__ __forceinline__ float GetInterpValueCUDA(
    const THCDeviceTensor<float, 3>& x, const THCDeviceTensor<float, 3>& obs,
    const float pos[3], const bool sample_into_geom) {

  // TODO(tompson,kris): THIS ASSUMES THAT OBSTACLES HAVE ZERO VELOCITY.

  // Make sure we're not against the grid boundary or beyond it.
  // This is a conservative test (i.e. we test if position is on or beyond it).
  assert(!IsOutOfDomainReal(pos, obs));

  // Get the current integer location of the pixel.
  int i0, j0, k0;
  GetPixelCenter(pos, &i0, &j0, &k0);
  assert(!IsOutOfDomain(i0, j0, k0, obs));

  // The current center SHOULD NOT be geometry.
  assert(!IsBlockedCell(obs, i0, j0, k0));

  const int dimsx = obs.getSize(2);
  const int dimsy = obs.getSize(1);
  const int dimsz = obs.getSize(0);

  // Calculate the next cell integer to interpolate with AND calculate the
  // interpolation coefficient.

  // If we're on the left hand size of the grid center we should be
  // interpolating left (p0) to center (p1), and if we're on the right we
  // should be interpolating center (p0) to right (p1).
  // RECALL: (0,0) is defined as the CENTER of the first cell. (xdim - 1,
  // ydim - 1) is defined as the center of the last cell.
  float icoef, jcoef, kcoef;
  int i1, j1, k1;
  if (pos[0] < (float)(i0)) {
    i1 = i0;
    i0 = max(i0 - 1, 0);
  } else {
    i1 = min(i0 + 1, dimsx - 1);
  }
  icoef = (i0 == i1) ? 0.0f : pos[0] - (float)(i0);

  // Same logic for top / bottom and front / back interp.
  if (pos[1] < (float)(j0)) {
    j1 = j0;
    j0 = max(j0 - 1, 0);
  } else {
    j1 = min(j0 + 1, dimsy - 1);
  }
  jcoef = (j0 == j1) ? 0.0f : pos[1] - (float)(j0);

  if (pos[2] < (float)(k0)) {
    k1 = k0;
    k0 = max(k0 - 1, 0);
  } else {
    k1 = min(k0 + 1, dimsz - 1);
  }
  kcoef = (k0 == k1) ? 0.0f : pos[2] - (float)(k0);

  // Mixing coefficients should be in [0, 1].
  assert(icoef >= 0 && icoef <= 1 && jcoef >= 0 && jcoef <= 1 &&
         kcoef >= 0 && kcoef <= 1);

  // Interpolation coordinates should be in the domain.
  assert(!IsOutOfDomain(i0, j0, k0, dims) && !IsOutOfDomain(i1, j1, k1, dims));

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
  const float xFrontLeftTop = x[k0][j1][i0];
  const bool gFrontLeftTop = IsBlockedCell(obs, i0, j1, k0);
  const float xFrontRightTop =  x[k0][j1][i1];
  const bool gFrontRightTop = IsBlockedCell(obs, i1, j1, k0);
  float xFrontTopInterp;
  bool gFrontTopInterp;
  MixWithGeomCUDA(xFrontLeftTop, xFrontRightTop, gFrontLeftTop, gFrontRightTop,
                  sample_into_geom, icoef, &xFrontTopInterp, &gFrontTopInterp);

  // Bottom Front MIX.
  const float xFrontLeftBottom = x[k0][j0][i0];
  const bool gFrontLeftBottom = IsBlockedCell(obs, i0, j0, k0);
  const float xFrontRightBottom = x[k0][j0][i1];
  const bool gFrontRightBottom = IsBlockedCell(obs, i1, j0, k0);
  float xFrontBottomInterp;
  bool gFrontBottomInterp;
  MixWithGeomCUDA(
    xFrontLeftBottom, xFrontRightBottom, gFrontLeftBottom, gFrontRightBottom,
    sample_into_geom, icoef, &xFrontBottomInterp, &gFrontBottomInterp);

  // Back face.
  // Top Back MIX.
  const float xBackLeftTop = x[k1][j1][i0];
  const bool gBackLeftTop = IsBlockedCell(obs, i0, j1, k1);
  const float xBackRightTop = x[k1][j1][i1];
  const bool gBackRightTop = IsBlockedCell(obs, i1, j1, k1);
  float xBackTopInterp;
  bool gBackTopInterp;
  MixWithGeomCUDA(xBackLeftTop, xBackRightTop, gBackLeftTop, gBackRightTop,
                  sample_into_geom, icoef, &xBackTopInterp, &gBackTopInterp);

  // Bottom Back MIX.
  const float xBackLeftBottom = x[k1][j0][i0];
  const bool gBackLeftBottom = IsBlockedCell(obs, i0, j0, k1);
  const float xBackRightBottom = x[k1][j0][i1];
  const bool gBackRightBottom = IsBlockedCell(obs, i1, j0, k1);
  float xBackBottomInterp;
  bool gBackBottomInterp;
  MixWithGeomCUDA(
      xBackLeftBottom, xBackRightBottom, gBackLeftBottom, gBackRightBottom,
      sample_into_geom, icoef, &xBackBottomInterp, &gBackBottomInterp);

  // Now get middle of front - The bilinear interp of the front face.
  float xBiLerpFront;
  bool gBiLerpFront;
  MixWithGeomCUDA(
      xFrontBottomInterp, xFrontTopInterp, gFrontBottomInterp, gFrontTopInterp,
      sample_into_geom, jcoef, &xBiLerpFront, &gBiLerpFront);

  // Now get middle of back - The bilinear interp of the back face.
  float xBiLerpBack;
  bool gBiLerpBack;
  MixWithGeomCUDA(
      xBackBottomInterp, xBackTopInterp, gBackBottomInterp, gBackTopInterp,
      sample_into_geom, jcoef, &xBiLerpBack, &gBiLerpBack);

  // Now get the interpolated point between the points calculated in the front
  // and back faces - The trilinear interp part.
  float xTriLerp;
  bool gTriLerp;
  MixWithGeomCUDA(xBiLerpFront, xBiLerpBack, gBiLerpFront, gBiLerpBack,
                  sample_into_geom, kcoef, &xTriLerp, &gTriLerp);

  // At least ONE of the samples shouldn't have been geometry so the final value
  // should be valid.
  assert(!gTriLerp);

  return xTriLerp;
}

__device__ __forceinline__ float sampleFieldCUDA(
    const THCDeviceTensor<float, 3>& field, const float pos[3],
    const THCDeviceTensor<float, 3>& geom, const bool sample_into_geom) {
  // This is a stupid wrap. But keep for argument parity with the C++ code.
  return GetInterpValueCUDA(field, geom, pos, sample_into_geom);
}

//******************************************************************************
// advectScalar
//******************************************************************************

// advectScalarEuler CUDA kernel.
__global__ void kernel_advectScalarEuler(
    const float dt, THCDeviceTensor<float, 3> p, THCDeviceTensor<float, 3> ux,
    THCDeviceTensor<float, 3> uy, THCDeviceTensor<float, 3> uz,
    THCDeviceTensor<float, 3> geom, const bool two_dim,
    THCDeviceTensor<float, 3> p_dst, const bool sample_into_geom) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = blockIdx.y;  // U-slice, i.e. Ux, Uy or Uz.
  const int k = blockIdx.z;
  if (i >= p.getSize(2)) {
    return;
  }

  if (geom[k][j][i] > 0.0f) {
    // Don't advect blocked cells.
    return;
  }

  // NOTE: The integer positions are in the center of the grid cells
  const float pos[3] = {(float)i, (float)j, (float)k};  // i.e. (x, y, z)
  
  // Velocity is in grids / second.
  const float vel[3] = {ux[k][j][i], uy[k][j][i],
                        two_dim ? 0.0f : uz[k][j][i]};
  
  // Backtrace based upon current velocity at cell center.
  const float displacement[3] = {-dt * vel[0], -dt * vel[1], -dt * vel[2]};
  float back_pos[3];
  // Step along the displacement vector. calcLineTrace will handle
  // boundary conditions for us. Note: it will terminate BEFORE the
  // boundary (i.e. the returned position is always valid).
  const bool hit_boundary = calcLineTraceCUDA(pos, displacement, geom,
                                              back_pos);
  
  // Check the return value from calcLineTrace just in case.
  assert(!IsOutOfDomainReal(back_pos, geom) &&
         !IsBlockedCellReal(obs, back_pos));
  
  // Finally, sample the value at the new position.
  p_dst[k][j][i] = sampleFieldCUDA(p, back_pos, geom, sample_into_geom);
} 

// advectScalarRK2 CUDA kernel.
__global__ void kernel_advectScalarRK2(
    const float dt, THCDeviceTensor<float, 3> p, THCDeviceTensor<float, 3> ux,
    THCDeviceTensor<float, 3> uy, THCDeviceTensor<float, 3> uz,
    THCDeviceTensor<float, 3> geom, const bool two_dim,
    THCDeviceTensor<float, 3> p_dst, const bool sample_into_geom) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = blockIdx.y;  // U-slice, i.e. Ux, Uy or Uz.
  const int k = blockIdx.z;
  if (i >= p.getSize(2)) {
    return;
  }

  if (geom[k][j][i] > 0.0f) {
    // Don't advect blocked cells.
    return;
  }

  // NOTE: The integer positions are in the center of the grid cells
  const float pos[3] = {(float)i, (float)j, (float)k};  // i.e. (x, y, z)

  // Velocity is in grids / second.
  float vel[3] = {ux[k][j][i], uy[k][j][i], two_dim ? 0.0f : uz[k][j][i]};

  // Backtrace a half step based upon current velocity at cell center.
  float displacement[3] = {0.5f * (-dt) * vel[0],
                           0.5f * (-dt) * vel[1],
                           0.5f * (-dt) * vel[2]};
  float half_pos[3];
  // Step along the displacement vector. calcLineTrace will handle
  // boundary conditions for us. Note: it will terminate BEFORE the
  // boundary (i.e. the returned position is always valid).
  const bool hit_boundary_half = calcLineTraceCUDA(pos, displacement, geom,
                                                   half_pos);

  // Check the return value from calcLineTrace just in case.
  assert(!IsOutOfDomainReal(half_pos, geom) &&
         !IsBlockedCellReal(obs, half_pos));

  if (hit_boundary_half) {
    // We hit the boundary, then as per Bridson, we should clamp the
    // backwards trace. Note: if we treated this as a full euler step, we 
    // would have hit the same blocker because the line trace is linear.
    // TODO(tompson,kris): I'm pretty sure this is the best we could do
    // but I still worry about numerical stability.
    p_dst[k][j][i] = sampleFieldCUDA(p, half_pos, geom, sample_into_geom);
    return;
  }

  // Sample the velocity at this half step location.
  // Note: dereferencing a 4D tensor returns a detail::THCDeviceSubTensor<...>
  // instance, NOT a THCDeviceTensor instance. We therefore need to create
  // new view instances (using the view function).
  vel[0] = sampleFieldCUDA(ux, half_pos, geom, true);
  vel[1] = sampleFieldCUDA(uy, half_pos, geom, true);
  vel[2] = two_dim ? 0.0f : sampleFieldCUDA(uz, half_pos, geom, true);

  // Do another line trace using this half position's velocity.
  float back_pos[3];
  displacement[0] = -dt * vel[0];
  displacement[1] = -dt * vel[1];
  displacement[2] = -dt * vel[2];
  const bool hit_boundary = calcLineTraceCUDA(pos, displacement, geom,
                                              back_pos);

  // Again, check the return value from calcLineTrace just in case.
  assert(!IsOutOfDomainReal(back_pos, geom) &&
         !IsBlockedCellReal(obs, back_pos));

  // Sample the value at the new position.
  p_dst[k][j][i] = sampleFieldCUDA(p, back_pos, geom, sample_into_geom);
}

// advectScalar lua entry point.
static int tfluids_CudaMain_advectScalar(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  const float dt = (float)(lua_tonumber(L, 1));
  THCudaTensor* p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* p_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  const std::string method = static_cast<std::string>(lua_tostring(L, 6));
  const bool sample_into_geom = static_cast<bool>(lua_toboolean(L, 7));

  if (p->nDimension != 3 || u->nDimension != 4 || geom->nDimension != 3 ||
      p_dst->nDimension != 3) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const bool two_dim = u->size[0] == 2;

  THCDeviceTensor<float, 3> dev_p = toDeviceTensor<float, 3>(state, p);
  THCDeviceTensor<float, 4> dev_u = toDeviceTensor<float, 4>(state, u);
  THCDeviceTensor<float, 3> dev_geom = toDeviceTensor<float, 3>(state, geom);
  THCDeviceTensor<float, 3> dev_p_dst = toDeviceTensor<float, 3>(state, p_dst);
  
  // One "thread" per output element.
  int nplane = dev_p_dst.getSize(2);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_p_dst.getSize(1),
                 dev_p_dst.getSize(0));  // (x, y, z)
  dim3 block_size(nplane > 256 ? 256 : nplane);

  // We have to dereference the 4D u tensor into x, y, and z components.
  // The reason for this is that dereferencing these in the kernel results in
  // THCDeviceSubTensor instances, which then cannot be passed to our sampling
  // function (which expects THCDeviceTensors only). Unfortunately, the view
  // method is host only, so we have to perform it here.
  THCDeviceTensor<float, 3> dev_ux = dev_u[0].view();
  THCDeviceTensor<float, 3> dev_uy = dev_u[1].view();
  // Note, if two_dim is true, we will set the z channel to an empty tensor.
  THCDeviceTensor<float, 3> dev_uz =
      two_dim ? THCDeviceTensor<float, 3>() : dev_u[2].view();

  if (method == "rk2") {
    kernel_advectScalarRK2<<<grid_size, block_size, 0,
                             THCState_getCurrentStream(state)>>>(
        dt, dev_p, dev_ux, dev_uy, dev_uz, dev_geom, two_dim, dev_p_dst,
        sample_into_geom);
  } else if (method == "euler") {
    kernel_advectScalarEuler<<<grid_size, block_size, 0,
                               THCState_getCurrentStream(state)>>>(
        dt, dev_p, dev_ux, dev_uy, dev_uz, dev_geom, two_dim, dev_p_dst,
        sample_into_geom);
  } else if (method == "maccormack") {
    luaL_error(L, "maccormack not yet implemented.");
  } else {
    luaL_error(L, "Invalid advection method.");
  }

  return 0;
}

//******************************************************************************
// advectVel
//******************************************************************************

// advectVelEuler CUDA kernel.
__global__ void kernel_advectVelEuler(
    const float dt, THCDeviceTensor<float, 3> ux,
    THCDeviceTensor<float, 3> uy, THCDeviceTensor<float, 3> uz,
    THCDeviceTensor<float, 3> geom, const bool two_dim,
    THCDeviceTensor<float, 3> ux_dst, THCDeviceTensor<float, 3> uy_dst,
    THCDeviceTensor<float, 3> uz_dst) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = blockIdx.y;  // U-slice, i.e. Ux, Uy or Uz.
  const int k = blockIdx.z;
  if (i >= ux.getSize(2)) {
    return;
  }
  if (geom[k][j][i] > 0.0f) {
    // Don't advect blocked cells.
    return;
  }
  // NOTE: The integer positions are in the center of the grid cells
  const float pos[3] = {(float)i, (float)j, (float)k};  // i.e. (x, y, z)
  
  // Velocity is in grids / second.
  const float vel[3] = {ux[k][j][i], uy[k][j][i],
                        two_dim ? 0.0f : uz[k][j][i]};
  
  // Backtrace based upon current velocity at cell center.
  const float displacement[3] = {-dt * vel[0], -dt * vel[1], -dt * vel[2]};
  float back_pos[3];
  // Step along the displacement vector. calcLineTrace will handle
  // boundary conditions for us. Note: it will terminate BEFORE the
  // boundary (i.e. the returned position is always valid).
  const bool hit_boundary = calcLineTraceCUDA(pos, displacement, geom,
                                              back_pos);
  
  // Check the return value from calcLineTrace just in case.
  assert(!IsOutOfDomainReal(back_pos, geom) &&
         !IsBlockedCellReal(obs, back_pos));
  
  // Finally, sample the value at the new position.
  ux_dst[k][j][i] = sampleFieldCUDA(ux, back_pos, geom, true);
  uy_dst[k][j][i] = sampleFieldCUDA(uy, back_pos, geom, true);
  if (!two_dim) {
    uz_dst[k][j][i] = sampleFieldCUDA(uz, back_pos, geom, true);
  }
} 

// advectVelRK2 CUDA kernel.
__global__ void kernel_advectVelRK2(
    const float dt, THCDeviceTensor<float, 3> ux,
    THCDeviceTensor<float, 3> uy, THCDeviceTensor<float, 3> uz,
    THCDeviceTensor<float, 3> geom, const bool two_dim,
    THCDeviceTensor<float, 3> ux_dst, THCDeviceTensor<float, 3> uy_dst, 
    THCDeviceTensor<float, 3> uz_dst) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = blockIdx.y;  // U-slice, i.e. Ux, Uy or Uz.
  const int k = blockIdx.z;
  if (i >= ux.getSize(2)) {
    return;
  }
  if (geom[k][j][i] > 0.0f) {
    // Don't advect blocked cells.
    return;
  }
  // NOTE: The integer positions are in the center of the grid cells
  const float pos[3] = {(float)i, (float)j, (float)k};  // i.e. (x, y, z)
  // Velocity is in grids / second.
  float vel[3] = {ux[k][j][i], uy[k][j][i], two_dim ? 0.0f : uz[k][j][i]};
  // Backtrace a half step based upon current velocity at cell center.
  float displacement[3] = {0.5f * (-dt) * vel[0],
                           0.5f * (-dt) * vel[1],
                           0.5f * (-dt) * vel[2]};
  float half_pos[3];
  // Step along the displacement vector. calcLineTrace will handle
  // boundary conditions for us. Note: it will terminate BEFORE the
  // boundary (i.e. the returned position is always valid).
  const bool hit_boundary_half = calcLineTraceCUDA(pos, displacement, geom,
                                                   half_pos);
  // Check the return value from calcLineTrace just in case.
  assert(!IsOutOfDomainReal(half_pos, geom) &&
         !IsBlockedCellReal(obs, half_pos));
  if (hit_boundary_half) {
    // We hit the boundary, then as per Bridson, we should clamp the
    // backwards trace. Note: if we treated this as a full euler step, we 
    // would have hit the same blocker because the line trace is linear.
    // TODO(tompson,kris): I'm pretty sure this is the best we could do
    // but I still worry about numerical stability.
    ux_dst[k][j][i] = sampleFieldCUDA(ux, half_pos, geom, true);
    uy_dst[k][j][i] = sampleFieldCUDA(uy, half_pos, geom, true);
    if (!two_dim) {
      uz_dst[k][j][i] = sampleFieldCUDA(uz, half_pos, geom, true);
    }
    return;
  }

  // Sample the velocity at this half step location.
  // Note: dereferencing a 4D tensor returns a detail::THCDeviceSubTensor<...>
  // instance, NOT a THCDeviceTensor instance. We therefore need to create
  // new view instances (using the view function).
  vel[0] = sampleFieldCUDA(ux, half_pos, geom, true);
  vel[1] = sampleFieldCUDA(uy, half_pos, geom, true);
  vel[2] = two_dim ? 0.0f : sampleFieldCUDA(uz, half_pos, geom, true);
  // Do another line trace using this half position's velocity.
  float back_pos[3];
  displacement[0] = -dt * vel[0];
  displacement[1] = -dt * vel[1];
  displacement[2] = -dt * vel[2];
  const bool hit_boundary = calcLineTraceCUDA(pos, displacement, geom,
                                              back_pos);
  // Again, check the return value from calcLineTrace just in case.
  assert(!IsOutOfDomainReal(back_pos, geom) &&
         !IsBlockedCellReal(obs, back_pos));
  // Sample the value at the new position.
  ux_dst[k][j][i] = sampleFieldCUDA(ux, back_pos, geom, true);
  uy_dst[k][j][i] = sampleFieldCUDA(uy, back_pos, geom, true);
  if (!two_dim) {
    uz_dst[k][j][i] = sampleFieldCUDA(uz, back_pos, geom, true);
  }
}

static int tfluids_CudaMain_advectVel(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  const float dt = (float)(lua_tonumber(L, 1));
  THCudaTensor* u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* geom = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* u_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  const std::string method = static_cast<std::string>(lua_tostring(L, 5));
  
  if (u->nDimension != 4 || geom->nDimension != 3 || u_dst->nDimension != 4) {
    luaL_error(L, "Incorrect dimensions.");
  }
  const bool two_dim = u->size[0] == 2;
  if (u->size[0] != u_dst->size[0]) {
    luaL_error(L, "u and u_dst size mismatch.");
  }

  THCDeviceTensor<float, 4> dev_u = toDeviceTensor<float, 4>(state, u);
  THCDeviceTensor<float, 3> dev_geom = toDeviceTensor<float, 3>(state, geom);
  THCDeviceTensor<float, 4> dev_u_dst = toDeviceTensor<float, 4>(state, u_dst);

  // One "thread" per output grid element (i.e. d * h * w).
  int nplane = dev_u_dst.getSize(3);
  dim3 grid_size(THCCeilDiv(nplane, 256), dev_u_dst.getSize(2),
                 dev_u_dst.getSize(1));  // (x, y, z)
  dim3 block_size(nplane > 256 ? 256 : nplane);

  // We have to dereference the 4D u tensors into x, y, and z components.
  // The reason for this is that dereferencing these in the kernel results in
  // THCDeviceSubTensor instances, which then cannot be passed to our sampling
  // function (which expects THCDeviceTensors only). Unfortunately, the view
  // method is host only, so we have to perform it here.
  THCDeviceTensor<float, 3> dev_ux = dev_u[0].view();
  THCDeviceTensor<float, 3> dev_uy = dev_u[1].view();
  // Note, if two_dim is true, we will set the z channel to an empty tensor.
  THCDeviceTensor<float, 3> dev_uz =
      two_dim ? THCDeviceTensor<float, 3>() : dev_u[2].view();
  THCDeviceTensor<float, 3> dev_ux_dst = dev_u_dst[0].view();
  THCDeviceTensor<float, 3> dev_uy_dst = dev_u_dst[1].view();
  THCDeviceTensor<float, 3> dev_uz_dst =
      two_dim ? THCDeviceTensor<float, 3>() : dev_u_dst[2].view();

  if (method == "rk2") {
    kernel_advectVelRK2<<<grid_size, block_size, 0,
                          THCState_getCurrentStream(state)>>>(
        dt, dev_ux, dev_uy, dev_uz, dev_geom, two_dim, dev_ux_dst,
        dev_uy_dst, dev_uz_dst);
  } else if (method == "euler") {
    kernel_advectVelEuler<<<grid_size, block_size, 0,
                            THCState_getCurrentStream(state)>>>(
        dt, dev_ux, dev_uy, dev_uz, dev_geom, two_dim, dev_ux_dst,
        dev_uy_dst, dev_uz_dst);
  } else if (method == "maccormack") {
    luaL_error(L, "maccormack not yet implemented.");
  } else {
    luaL_error(L, "Invalid advection method.");
  }

  return 0;  
}

//******************************************************************************
// INIT METHODS
//******************************************************************************

static const struct luaL_Reg tfluids_CudaMain__ [] = {
  {"averageBorderCells", tfluids_CudaMain_averageBorderCells},
  {"setObstacleBcs", tfluids_CudaMain_setObstacleBcs},
  {"vorticityConfinement", tfluids_CudaMain_vorticityConfinement},
  {"calcVelocityUpdate", tfluids_CudaMain_calcVelocityUpdate},
  {"calcVelocityUpdateBackward", tfluids_CudaMain_calcVelocityUpdateBackward},
  {"calcVelocityDivergence", tfluids_CudaMain_calcVelocityDivergence},
  {"calcVelocityDivergenceBackward",
   tfluids_CudaMain_calcVelocityDivergenceBackward},
  {"solveLinearSystemPCG", tfluids_CudaMain_solveLinearSystemPCG},
  {"advectScalar", tfluids_CudaMain_advectScalar},
  {"advectVel", tfluids_CudaMain_advectVel},
  {NULL, NULL}  // NOLINT
};

const struct luaL_Reg* tfluids_CudaMain_getMethodsTable() {
  return tfluids_CudaMain__;
}

void tfluids_CudaMain_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, tfluids_CudaMain__, "tfluids");
}
