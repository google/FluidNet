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

#include "luaT.h"
#include "THC.h"

#include <float.h>
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"

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
    out[chan][z][y][x] = sum / static_cast<float>(count);
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
  const int32_t nchan = in->size[0];
  const int32_t zdim = in->size[1];
  const int32_t ydim = in->size[2];
  const int32_t xdim = in->size[3];
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
  const int32_t zdim = U->size[1];
  const int32_t ydim = U->size[2];
  const int32_t xdim = U->size[3];
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

__device__ void getCurl3D(const THCDeviceTensor<float, 4>& U, const int i,
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

__device__ float getCurl2D(const THCDeviceTensor<float, 4>& U, const int i,
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

  const float dt = static_cast<float>(lua_tonumber(L, 1));
  const float scale = static_cast<float>(lua_tonumber(L, 2));  
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

  const int32_t nbatch = delta_u->size[0];
  const int32_t nuchan = delta_u->size[1];
  const int32_t zdim = delta_u->size[2];
  const int32_t ydim = delta_u->size[3];
  const int32_t xdim = delta_u->size[4];

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

  const int32_t nbatch = grad_p->size[0];
  const int32_t zdim = grad_p->size[1];
  const int32_t ydim = grad_p->size[2];
  const int32_t xdim = grad_p->size[3];
  const int32_t nuchan = grad_output->size[1];

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
  for (int32_t dim = 0; dim < nuchan; dim ++) {
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
  const int32_t nbatch = u->size[0];
  const int32_t nuchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];

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
  for (int32_t dim = 0; dim < nuchan; dim ++) {
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
  const int32_t nbatch = u->size[0];
  const int32_t nuchan = u->size[1];
  const int32_t zdim = u->size[2];
  const int32_t ydim = u->size[3];
  const int32_t xdim = u->size[4];

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

static const struct luaL_Reg tfluids_CudaMain__ [] = {
  {"averageBorderCells", tfluids_CudaMain_averageBorderCells},
  {"setObstacleBcs", tfluids_CudaMain_setObstacleBcs},
  {"vorticityConfinement", tfluids_CudaMain_vorticityConfinement},
  {"calcVelocityUpdate", tfluids_CudaMain_calcVelocityUpdate},
  {"calcVelocityUpdateBackward", tfluids_CudaMain_calcVelocityUpdateBackward},
  {"calcVelocityDivergence", tfluids_CudaMain_calcVelocityDivergence},
  {"calcVelocityDivergenceBackward",
   tfluids_CudaMain_calcVelocityDivergenceBackward},
  {NULL, NULL}  // NOLINT
};

void tfluids_CudaMain_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, tfluids_CudaMain__, "tfluids");
}
