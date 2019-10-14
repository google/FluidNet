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
#include <cublas_v2.h>
#include <float.h>
#include <algorithm>
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"

#include "generic/advect_type.h"
#include "third_party/cell_type.h"
#include "third_party/grid.cu.h"
#include "generic/int3.cu.h"
#include "generic/vec3.cu.h"

// The PCG code also does some processing on the CPU, and so we need the
// headers for grid, vec3, etc.
#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define tfluids_(NAME) TH_CONCAT_3(tfluids_, Real, NAME)
#define real float
#define accreal double
#define Real Float
#define THInf FLT_MAX
#define TH_REAL_IS_FLOAT
#include "generic/vec3.h"
#include "third_party/grid.h"
#include "generic/find_connected_fluid_components.h"
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_FLOAT

#include "generic/calc_line_trace.cu"

const int threads_per_block = 512;  // Might need 256 for old SM.
const int64_t cuda_num_threads = 1024;  // Might need 256 for old SM.

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

// *****************************************************************************
// LaunchKernel
// *****************************************************************************

// A simple helper function to reduce the amount of boiler plate code required
// to launch a kernel (it also cuts down the number of potential bugs).
// 
// All our kernels use an unknown number of parameters, so we'll need to
// pass in a function pointer with the correct signature as well as the
// arg lists.
//
// @template TFuncPtr: kernel func ptr. The compiler will autocomplete this!
// @template Args: Again, you do not need to define it (see emptyDomain).
// @param: func - the kernel function to call.
// @param: <x>size - The size of the domain that the kernel will be launched
// over.  This MUST match the domain used in GetKernelIndices.
// @param: args - the variable size argument list that the kernel takes as
// input.
template <typename TFuncPtr, typename... Args>  // C++11 varadic function
static void LaunchKernel(lua_State* L, TFuncPtr func,
                         const int bsize, const int csize, const int zsize,
                         const int ysize, const int xsize,
                         Args... args) {
  THCState* state = cutorch_getstate(L);

  // Create the kernel grid and block sizes.
  // TODO(tompson): What if csize is 1 (i.e. scalar domains). Is this slower?
  int nplane = xsize * ysize * zsize;
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), csize, bsize);
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);

  // Call the function.
  func<<<grid_size, block_size, 0, THCState_getCurrentStream(state)>>>(args...);
}

// Same as above, but on a one of our Grid objects.
template <typename TFuncPtr, typename... Args>  // C++11 varadic function
static void LaunchKernel(lua_State* L, TFuncPtr func,
                         const CudaGridBase& domain, Args... args) {
  THCState* state = cutorch_getstate(L);
  const int xsize = domain.xsize();
  const int ysize = domain.ysize();
  const int zsize = domain.zsize();
  const int csize = domain.nchan();
  const int bsize = domain.nbatch();

  // Create the kernel grid and block sizes.
  // TODO(tompson): What if csize is 1 (i.e. scalar domains). Is this slower?
  int nplane = xsize * ysize * zsize;
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), csize, bsize);
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);

  // Call the function.
  func<<<grid_size, block_size, 0, THCState_getCurrentStream(state)>>>(args...);
  THCudaCheck(cudaGetLastError());
}

inline int64_t GetBlocks(const int64_t n) {
  return (n + cuda_num_threads - 1) / cuda_num_threads;
}

// This method will launch a kernel over the entire domain numel.
template <typename TFuncPtr, typename... Args>  // C++11 varadic function
static void LaunchKernelLoop(lua_State* L, TFuncPtr func,
                             const CudaGridBase& domain, Args... args) {
  THCState* state = cutorch_getstate(L);

  // Call the function.
  // const int64_t numel = THCudaTensor_nElement(state, domain);
  const int64_t numel = domain.numel();
  func<<<GetBlocks(numel), cuda_num_threads, 0,
         THCState_getCurrentStream(state)>>>(args...);
  THCudaCheck(cudaGetLastError());
}

// Assumes you're iterating over a scalar domain (i.e nchan = 1 for the domain
// you're iterating over). The LaunchKernelLoop forces this since you cannot
// specify a nchan.
__device__ __forceinline__ void PntIdToScalarIndices(
    const int32_t nbatch, const int32_t zsize, const int32_t ysize,
    const int32_t xsize, const int32_t& pnt_id, int32_t& batch,
    int32_t& k, int32_t& j, int32_t& i) {
  i = pnt_id % xsize;
  j = (pnt_id / xsize) % ysize;
  k = (pnt_id / xsize / ysize) % zsize;
  batch = (pnt_id / xsize / ysize / zsize);
}

// CUDA: grid stride looping.
// This strategy comes from similar code in the cunn library.
#define CUDA_KERNEL_LOOP(numel, pnt_id) \
  for (int32_t pnt_id = blockIdx.x * blockDim.x + threadIdx.x; \
       pnt_id < (numel); \
       pnt_id += blockDim.x * gridDim.x)

// *****************************************************************************
// GetKernelIndices
// *****************************************************************************

// Another helper function to get back the batch, chan, k, j, i indices in a
// kernel launch by the LaunchKernel function above.
//
// If GetKernelIndices returns true, then the current kernel is out of the
// domain (and so you should just exist the kernel). This happens because
// the tensor may not fill up the last grid.
//
// Note, you should ALWAYS pass in the same sizes as the tensor you used
// to call the kernel in LaunchKernel's domain parameter.
__device__ __forceinline__ bool GetKernelIndices(
    const int32_t bsize, const int32_t csize, const int32_t zsize,
    const int32_t ysize, const int32_t xsize, int32_t& batch, int32_t& chan,
    int32_t& k, int32_t& j, int32_t& i) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  chan = blockIdx.y;
  batch = blockIdx.z;
  if (pnt_id >= zsize * ysize * xsize) {
    return true;
  }
  i = pnt_id % xsize;
  j = (pnt_id / xsize) % ysize;
  k = pnt_id / (xsize * ysize);
  return false;
}

// Same as above but on one of our Grid objects.
__device__ __forceinline__ bool GetKernelIndices(
    const CudaGridBase& domain, int32_t& batch, int32_t& chan, int32_t& k,
    int32_t& j, int32_t& i) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  chan = blockIdx.y;
  batch = blockIdx.z;
  if (pnt_id >= (domain.zsize() * domain.ysize() * domain.xsize())) {
    return true;
  }
  i = pnt_id % domain.xsize();
  j = (pnt_id / domain.xsize()) % domain.ysize();
  k = pnt_id / (domain.ysize() * domain.xsize());
  return false;
}

// There are a LOT of methods in tfluids that borrow heavily (or port) parts of
// Manta. These are compiled here but note that they are added under a separate
// license. You should see FluidNet/torch/tfluids/third_party/README for more
// information.
#include "third_party/tfluids.cu"

// *****************************************************************************
// velocityDivergenceBackward
// *****************************************************************************

__global__ void velocityDivergenceBackward(
    CudaFlagGrid flags, CudaMACGrid grad_u, CudaRealGrid grad_output,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border in the forward pass, so they do
    // not contribute gradient.
    return;
  }

  if (!flags.isFluid(i, j, k, b)) {
    // Blocked cells don't contribute gradient.
    return;
  }

  // TODO(tompson): I'm sure these atomic add calls are slow! We should
  // probably change this from a scatter to a gather op to avoid having to use
  // them at all.
  // (NVIDIA state that atomic operations on global memory are extremely slow)
  // but on shared memory it is OK. So we could copy to shared first, use
  // atomic ops there then use a small number of atomic ops back to global mem
  // (probably rewriting it as a gather would be easier).
  const float go = grad_output(i, j, k, b);
  atomicAdd(&grad_u(i, j, k, 0, b), go);
  atomicAdd(&grad_u(i + 1, j, k, 0, b), -go);
  atomicAdd(&grad_u(i, j, k, 1, b), go);
  atomicAdd(&grad_u(i, j + 1, k, 1, b), -go); 
  if (flags.is_3d()) {
    atomicAdd(&grad_u(i, j, k, 2, b), go);
    atomicAdd(&grad_u(i, j, k + 1, 2, b), -go); 
  } 
}

static int tfluids_CudaMain_velocityDivergenceBackward(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));
  THCudaTensor* tensor_grad_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid grad_u = toCudaMACGrid(state, tensor_grad_u, is_3d);
  CudaRealGrid grad_output = toCudaRealGrid(state, tensor_grad_output, is_3d);

  // Firstly, we're going to accumulate gradient contributions, so set
  // grad_u to 0.
  THCudaTensor_zero(state, tensor_grad_u);

  // LaunchKernel args: lua_State, func, domain, args...
  const int32_t bnd = 1;
  LaunchKernel(L, &velocityDivergenceBackward, flags,
               flags, grad_u, grad_output, bnd);
   
  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// emptyDomain
// *****************************************************************************

__global__ void emptyDomainLoop(
    CudaFlagGrid flags, const bool is_3d, const int32_t bnd,
    const int32_t nbatch, const int32_t zsize, const int32_t ysize,
    const int32_t xsize, const int32_t numel) {
  int32_t b, k, j, i;
  CUDA_KERNEL_LOOP(numel, pnt_id) {
    PntIdToScalarIndices(nbatch, zsize, ysize, xsize, pnt_id, b, k, j, i);  
    if (i < bnd || i > flags.xsize() - 1 - bnd ||
        j < bnd || j > flags.ysize() - 1 - bnd ||
        (is_3d && (k < bnd || k > flags.zsize() - 1 - bnd))) {
      flags(i, j, k, b) = TypeObstacle;
    } else {
      flags(i, j, k, b) = TypeFluid;
    }
  }
}

__global__ void emptyDomain(
     CudaFlagGrid flags, const bool is_3d, const int32_t bnd) {
  int32_t b, dim, k, j, i;
  if (GetKernelIndices(flags, b, dim, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (is_3d && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    flags(i, j, k, b) = TypeObstacle;
  } else {
    flags(i, j, k, b) = TypeFluid;
  }
}

static int tfluids_CudaMain_emptyDomain(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 2));
  const int32_t bnd = static_cast<int32_t>(lua_tointeger(L, 3));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  // Looped version - Actually not really any faster..
  // LaunchKernelLoop(L, &emptyDomainLoop, flags,
  //                  flags, is_3d, bnd, flags.nbatch(), flags.zsize(),
  //                  flags.ysize(), flags.xsize(), flags.numel());
  LaunchKernel(L, &emptyDomain, flags,
               flags, is_3d, bnd);
  return 0;
}

// *****************************************************************************
// flagsToOccupancy
// *****************************************************************************
__global__ void flagsToOccupancy(CudaFlagGrid flags,
                                 CudaFlagGrid occupancy) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  float val;
  if (flags.isFluid(i, j, k, b)) {
    val = 0;
  } else if (flags.isObstacle(i, j, k, b)) {
    val = 1;
  } else {
    val = -1;  // Can't throw error in kernel. Set to -1 and check min.
  }
  occupancy(i, j, k, b) = val;
}

static int tfluids_CudaMain_flagsToOccupancy(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_occupancy = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));

  // Normally, we would pass this in, but actually it doesn't make a difference
  // to the calculation.
  const bool is_3d = tensor_flags->size[2] > 1;

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaFlagGrid occupancy = toCudaFlagGrid(state, tensor_occupancy, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &flagsToOccupancy, flags,
               flags, occupancy);

  // We could be pedantic and check that the occupancy grid is OK. But this
  // reduction is very expensive on GPU.
  // if (THCudaTensor_minall(state, tensor_occupancy) < 0) {
  //   luaL_error(L, "ERROR: unsupported flag cell found!");
  // } 

  return 0;
}

// *****************************************************************************
// velocityUpdateBackward
// *****************************************************************************

__global__ void velocityUpdateBackward(
    CudaFlagGrid flags, CudaMACGrid grad_output, CudaRealGrid grad_p,
    const int32_t bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }
  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    // Manta zeros stuff on the border in the forward pass, so they do
    // not contribute gradient.
    return;
  }
  const CudaVec3 go(grad_output(i, j, k, b));

  // TODO(tompson): I'm sure these atomic add calls are slow! We should
  // probably change this from a scatter to a gather op to avoid having to use
  // them at all.
  // (NVIDIA state that atomic operations on global memory are extremely slow)
  // but on shared memory it is OK. So we could copy to shared first, use
  // atomic ops there then use a small number of atomic ops back to global mem
  // (probably rewriting it as a gather would be easier).
  if (flags.isFluid(i, j, k, b)) {
    if (flags.isFluid(i - 1, j, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.x);
      atomicAdd(&grad_p(i - 1, j, k, b), go.x);
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.y);
      atomicAdd(&grad_p(i, j - 1, k, b), go.y);
    }
    if (flags.is_3d() && flags.isFluid(i, j, k - 1, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.z); 
      atomicAdd(&grad_p(i, j, k - 1, b), go.z);
    }

    if (flags.isEmpty(i - 1, j, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.x);
    }
    if (flags.isEmpty(i, j - 1, k, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.y);
    }
    if (flags.is_3d() && flags.isEmpty(i, j, k - 1, b)) {
      atomicAdd(&grad_p(i, j, k, b), -go.z);
    }
  }
  else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
    // don't change velocities in outflow cells   
    if (flags.isFluid(i - 1, j, k, b)) {
      atomicAdd(&grad_p(i - 1, j, k, b), go.x);
    } else {
      // Output doesn't depend on p, so gradient is zero and so doesn't
      // contribute.
    }
    if (flags.isFluid(i, j - 1, k, b)) {
      atomicAdd(&grad_p(i, j - 1, k, b), go.y);
    } else {
      // Output doesn't depend on p, so gradient is zero and so doesn't
      // contribute.
    }
    if (flags.is_3d()) {
      if (flags.isFluid(i, j, k - 1, b)) {
        atomicAdd(&grad_p(i, j, k - 1, b), go.z);
      } else {
        // Output doesn't depend on p, so gradient is zero and so
        // doesn't contribute.
      }
    }
  }
}

static int tfluids_CudaMain_velocityUpdateBackward(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_u = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 5));
  THCudaTensor* tensor_grad_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaMACGrid grad_output = toCudaMACGrid(state, tensor_grad_output, is_3d);
  CudaRealGrid grad_p = toCudaRealGrid(state, tensor_grad_p, is_3d);

  // Firstly, we're going to accumulate gradient contributions, so set
  // grad_p to 0.
  THCudaTensor_zero(state, tensor_grad_p);

  const int32_t bnd = 1;
  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &velocityUpdateBackward, flags,
               flags, grad_output, grad_p, bnd);

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// volumetricUpsamplingNearestForward
// *****************************************************************************

__global__ void volumetricUpSamplingNearestForward(
    const int ratio, THCDeviceTensor<float, 5> in,
    THCDeviceTensor<float, 5> out) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int chan = blockIdx.y;
  const int batch = blockIdx.z;
  if (pnt_id >= (out.getSize(2) * out.getSize(3) * out.getSize(4))) {
    return;
  }
  const int x = pnt_id % out.getSize(4);
  const int y = (pnt_id / out.getSize(4)) % out.getSize(3);
  const int z = pnt_id / (out.getSize(3) * out.getSize(4));

  const int xin = x / ratio;
  const int yin = y / ratio;
  const int zin = z / ratio;
  const float inVal = in[batch][chan][zin][yin][xin];
  out[batch][chan][z][y][x] = inVal;
}

static int tfluids_CudaMain_volumetricUpSamplingNearestForward(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THCudaTensor* input = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));

  if (input->nDimension != 5 || output->nDimension != 5) {
    luaL_error(L, "ERROR: input and output must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zdim = input->size[2];
  const int32_t ydim = input->size[3];
  const int32_t xdim = input->size[4];

  if (output->size[0] != nbatch || output->size[1] != nfeat ||
      output->size[2] != zdim * ratio || output->size[3] != ydim * ratio ||
      output->size[4] != xdim * ratio) {
    luaL_error(L, "ERROR: input : output size mismatch.");
  }

  THCDeviceTensor<float, 5> dev_in = toDeviceTensor<float, 5>(state, input);
  THCDeviceTensor<float, 5> dev_out = toDeviceTensor<float, 5>(state, output);

  if (!THCudaTensor_isContiguous(state, input)) {
    luaL_error(L, "ERROR: input must be contiguous");
  }
  if (!THCudaTensor_isContiguous(state, output)) {
    luaL_error(L, "ERROR: output must be contiguous");
  }

  // One thread per output element.
  int nplane = dev_out.getSize(2) * dev_out.getSize(3) * dev_out.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), dev_out.getSize(1),
                 dev_out.getSize(0));
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);

  volumetricUpSamplingNearestForward<<<grid_size, block_size, 0,
                                       THCState_getCurrentStream(state)>>>(
      ratio, dev_in, dev_out);

  return 0;
}

// *****************************************************************************
// volumetricUpsamplingNearestBackward
// *****************************************************************************

__global__ void volumetricUpSamplingNearestBackward(
    const int ratio, THCDeviceTensor<float, 5> grad_out,
    THCDeviceTensor<float, 5> grad_in) {
  const int pnt_id = threadIdx.x + blockIdx.x * blockDim.x;
  const int chan = blockIdx.y;
  const int batch = blockIdx.z; 
  if (pnt_id >= (grad_in.getSize(2) * grad_in.getSize(3) *
      grad_in.getSize(4))) {
    return;
  }
  const int x = pnt_id % grad_in.getSize(4);
  const int y = (pnt_id / grad_in.getSize(4)) % grad_in.getSize(3);
  const int z = pnt_id / (grad_in.getSize(3) * grad_in.getSize(4));
 
  float sum = 0.0f;

  // Now accumulate gradients from the upsampling window.
  for (int32_t zup = 0; zup < ratio; zup++) { 
    for (int32_t yup = 0; yup < ratio; yup++) { 
      for (int32_t xup = 0; xup < ratio; xup++) {
        const int xin = x * ratio + xup;
        const int yin = y * ratio + yup;
        const int zin = z * ratio + zup;
        const float val = grad_out[batch][chan][zin][yin][xin];
        sum += val;
      }
    }
  }
        
  grad_in[batch][chan][z][y][x] = sum;
}

static int tfluids_CudaMain_volumetricUpSamplingNearestBackward(lua_State* L) {
  THCState* state = cutorch_getstate(L);
  
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THCudaTensor* input = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* grad_output = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* grad_input = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  
  if (input->nDimension != 5 || grad_output->nDimension != 5 ||
      grad_input->nDimension != 5) {
    luaL_error(L, "ERROR: input, gradOutput and gradInput must be dim 5");
  }
  
  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zdim = input->size[2];
  const int32_t ydim = input->size[3];
  const int32_t xdim = input->size[4];

  if (grad_output->size[0] != nbatch || grad_output->size[1] != nfeat ||
      grad_output->size[2] != zdim * ratio ||
      grad_output->size[3] != ydim * ratio ||
      grad_output->size[4] != xdim * ratio) {
    luaL_error(L, "ERROR: input : gradOutput size mismatch.");
  }

  if (grad_input->size[0] != nbatch || grad_input->size[1] != nfeat ||
      grad_input->size[2] != zdim || grad_input->size[3] != ydim ||
      grad_input->size[4] != xdim) {
    luaL_error(L, "ERROR: input : gradInput size mismatch.");
  }

  THCDeviceTensor<float, 5> dev_in = toDeviceTensor<float, 5>(state, input);
  THCDeviceTensor<float, 5> dev_grad_out = toDeviceTensor<float, 5>(
      state, grad_output);
  THCDeviceTensor<float, 5> dev_grad_in = toDeviceTensor<float, 5>(
    state, grad_input);
  
  if (!THCudaTensor_isContiguous(state, input)) {
    luaL_error(L, "ERROR: input must be contiguous");
  }
  if (!THCudaTensor_isContiguous(state, grad_output)) {
    luaL_error(L, "ERROR: gradOutput must be contiguous");
  }
  if (!THCudaTensor_isContiguous(state, grad_input)) {
    luaL_error(L, "ERROR: gradInput must be contiguous");
  }

  // One thread per grad_input element.
  // TODO(tompson): This is slow. Switch to a looping kernel.
  int nplane = dev_grad_in.getSize(2) * dev_grad_in.getSize(3) *
    dev_grad_in.getSize(4);
  dim3 grid_size(THCCeilDiv(nplane, threads_per_block), dev_grad_in.getSize(1),
                 dev_grad_in.getSize(0));  
  dim3 block_size(nplane > threads_per_block ? threads_per_block : nplane);
  
  volumetricUpSamplingNearestBackward<<<grid_size, block_size, 0,
                                        THCState_getCurrentStream(state)>>>(
      ratio, dev_grad_out, dev_grad_in);

  return 0;
}

// *****************************************************************************
// signedDistanceField
// *****************************************************************************

__global__ void signedDistanceField(
    CudaFlagGrid flags, const int search_rad, CudaRealGrid dst) {
  int b, chan, z, y, x;
  if (GetKernelIndices(flags, b, chan, z, y, x)) {
    return;
  }

  if (flags.isObstacle(x, y, z, b)) {
    dst(x, y, z, b) = 0;
  }

  float dist_sq = static_cast<float>(search_rad * search_rad);
  const int zmin = max(0, z - search_rad);;
  const int zmax = min((int)flags.zsize() - 1, z + search_rad);
  const int ymin = max(0, y - search_rad);;
  const int ymax = min((int)flags.ysize() - 1, y + search_rad);
  const int xmin = max(0, x - search_rad);;
  const int xmax = min((int)flags.xsize() - 1, x + search_rad);
  for (int zsearch = zmin; zsearch <= zmax; zsearch++) {
    for (int ysearch = ymin; ysearch <= ymax; ysearch++) {
      for (int xsearch = xmin; xsearch <= xmax; xsearch++) {
        if (flags.isObstacle(xsearch, ysearch, zsearch, b)) {
          const float cur_dist_sq = ((z - zsearch) * (z - zsearch) +
                                     (y - ysearch) * (y - ysearch) +
                                     (x - xsearch) * (x - xsearch));
          if (dist_sq > cur_dist_sq) {
            dist_sq = cur_dist_sq;
          }
        }
      }
    }
  }
  dst(x, y, z, b) = sqrt(dist_sq);
}

static int tfluids_CudaMain_signedDistanceField(lua_State *L) {
  THCState* state = cutorch_getstate(L);

  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  const int32_t search_rad = static_cast<int32_t>(lua_tointeger(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));
  THCudaTensor* tensor_dst = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaRealGrid dst = toCudaRealGrid(state, tensor_dst, is_3d);

  // LaunchKernel args: lua_State, func, domain, args...
  LaunchKernel(L, &signedDistanceField, flags,
               flags, search_rad, dst);

  return 0;
}

//******************************************************************************
// solveLinearSystemPCG
//******************************************************************************

static cublasHandle_t cublas_handle = 0;

static void init_cublas() {
  if (cublas_handle == 0) {
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      THError("CUBLAS Library initialization failed");
    }
  }
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

// Method from:
// stackoverflow.com/questions/30454089/solving-sparse-definite-positive-linear-systems-in-cuda  // NOLINT
static const char* cusparseGetStatusString(cusparseStatus_t status) {
  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "CUSPARSE_STATUS_ZERO_PIVOT";
    default:
      return "<unknown cusparse error>";
  }
}

#define CHECK_CUSPARSE(expr) checkCusparseStatus((expr), __FILE__, __LINE__)

void checkCusparseStatus(cusparseStatus_t stat, char const * file, int line) {
  if (stat != CUSPARSE_STATUS_SUCCESS) {
    std::cout << "CUSPARSE error in file '" << file << "', line " << line
              << ": error(" << stat << "): "
              << cusparseGetStatusString(stat) << std::endl;
  }
  THCudaCheck(cudaGetLastError());
  // Sometimes, torch's cuda handle wont catch the error but cusparse enum
  // is bad. If that's the case, hard fail here.
  if (stat != CUSPARSE_STATUS_SUCCESS) {
    THError("CUSPARSE error");
    exit(-1);
  }
}

static const char* cublasGetStatusString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:  
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default: 
      return "<unknown cublas error>";
  } 
}  

#define CHECK_CUBLAS(expr) checkCublasStatus((expr), __FILE__, __LINE__)

void checkCublasStatus(cublasStatus_t stat, char const * file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS error in file '" << file << "', line " << line
              << ": error(" << stat << "): "
              << cublasGetStatusString(stat) << std::endl;
  }
  THCudaCheck(cudaGetLastError());
  // Sometimes, torch's cuda handle wont catch the error but cusparse enum
  // is bad. If that's the case, hard fail here.
  if (stat != CUBLAS_STATUS_SUCCESS) {
    THError("CUBLAS error");
    exit(-1);
  }
}

// These macros require that state be defined, which you can get by calling
// cutorch_getstate.
#define DEV_PTR(tensor) THCudaTensor_data(state, tensor)
#define DEV_INT_PTR(tensor) THCudaIntTensor_data(state, tensor)

int64_t createReducedSystemIndices(
    const tfluids_FloatFlagGrid& flags, THIntTensor* components,
    THIntTensor* indices, const int32_t ibatch, const int32_t icomponent) {
  if (indices->nDimension != 3) {
    THError("indices must be 3D");
  }
  if (components->nDimension != 3) {
    THError("components must be 3D");
  }

  int64_t cur_index = 0;
  const int32_t zsize = flags.zsize();
  const int32_t ysize = flags.ysize();
  const int32_t xsize = flags.xsize();

  if ((indices->size[0] != zsize || indices->size[1] != ysize ||
       indices->size[2] != xsize)) {
    THError("indices must be the same dimension as flags (non-batched)");
  }

  for (int32_t k = 0; k < zsize; k++) {
    for (int32_t j = 0; j < ysize; j++) {
      for (int32_t i = 0; i < xsize; i++) {
        if (THIntTensor_get3d(components, k, j, i) == icomponent) { 
          // Note, if it's part of the connected component of fluid cells then
          // we don't have to check if it's fluid. However we should do anyway
          // just to make sure.
          if (!flags.isFluid(i, j, k, ibatch)) {
            THError("A non fluid component was found!");
          }
          THIntTensor_set3d(indices, k, j, i, cur_index);
          cur_index++;
        } else {
          THIntTensor_set3d(indices, k, j, i, -1);
        }
      }
    }
  }

  return cur_index;
}

// @param I, col, and val: the output CSR formatted sparse matrix.
// TOTO(tompson): This is super slow. All the _get and _set methods do bounds
// checking. Once everything is working, switch to raw ptrs.
static int64_t setupLaplacian(
    const tfluids_FloatFlagGrid& flags, const int32_t b, THIntTensor* row,
    THIntTensor* col, THFloatTensor *val, const bool upper_triangular,
    THIntTensor* system_indices, THIntTensor* components,
    const int icomponent) {
  // row stores the indices of the first non-zero item in the col and val
  // arrays. (i.e. col[row[n]] is the column index of the 1st element of row n.
  // and val[row[4]] is the corresponding value.
  // The number of non-zero values in each row is given by row[n + 1] - row[n].
  // Hence the need to have (dim + 1) row values.
  int64_t current_row = 0;
  int64_t val_index = 0;
  
  THIntTensor_set1d(row, current_row, 0);  // 0th row starts at 0th index.

  // TODO(tompson): Parallelize this.
  const int32_t zsize = flags.zsize();
  const int32_t ysize = flags.ysize();
  const int32_t xsize = flags.xsize();
  const int32_t bnd = 1;
  for (int32_t k = 0; k < zsize; k++) {
    for (int32_t j = 0; j < ysize; j++) {
      for (int32_t i = 0; i < xsize; i++) {
        if (THIntTensor_get3d(components, k, j, i) != icomponent) {
          // Not part of the current connected component.
          // Make sure the current cell wasn't assigned an index in the output
          // system.
          if (THIntTensor_get3d(system_indices, k, j, i) != -1) {
            THError("Non fluid cell shouldn't have an index!");
          }
          continue;
        }
        const bool out_of_bounds =
            (i < bnd || i > xsize - 1 - bnd ||
             j < bnd || j > ysize - 1 - bnd ||
             (flags.is_3d() && (k < bnd || k > zsize - 1 - bnd)));

        // As per Manta's convention, the border are all obstacle cells.
        // Therefore their divergence (rhs) is zero. AND the do not contribute
        // non-zero elements to the sparse matrix. As such, we just skip
        // over them.
       
        // Technically the isFluid check here is completely redundant (since
        // it's part of a component), but lets do it anyway for clarity).
        if (!out_of_bounds && flags.isFluid(i, j, k, b)) {
          // Collect the diagonal term first. The diag term is the sum of
          // NON obstacle cells. In most cases this is the same as fluid cells,
          // but empty cells also contribute flow.
          float val_diagonal = 0.0f;
          if (!flags.isObstacle(i - 1, j, k, b)) {
            val_diagonal += 1;
          }
          if (!flags.isObstacle(i + 1, j, k, b)) {
            val_diagonal += 1;
          }
          if (!flags.isObstacle(i, j - 1, k, b)) {
            val_diagonal += 1;
          }
          if (!flags.isObstacle(i, j + 1, k, b)) {
            val_diagonal += 1;
          }
          if (flags.is_3d() && !flags.isObstacle(i, j, k - 1, b)) {
            val_diagonal += 1;
          }
          if (flags.is_3d() && !flags.isObstacle(i, j, k + 1, b)) {
            val_diagonal += 1;
          }

          // Off diagonal entries.
          float im1jk = 0.0f;
          if (!upper_triangular && flags.isFluid(i - 1, j, k, b)) {
            im1jk = -1.0f;  // Off diagonal entry for fluid neighbors is -1.
          }
          float ip1jk = 0.0f;
          if (flags.isFluid(i + 1, j, k, b)) {
            ip1jk = -1.0f;
          }
          float ijm1k = 0.0f;
          if (!upper_triangular && flags.isFluid(i, j - 1, k, b)) {
            ijm1k = -1.0f;
          }
          float ijp1k = 0.0f;
          if (flags.isFluid(i, j + 1, k, b)) {
            ijp1k = -1.0f;
          }
          float ijkm1 = 0.0f;
          float ijkp1 = 0.0f;
          if (flags.is_3d()) {
            if (!upper_triangular && flags.isFluid(i, j, k - 1, b)) {
              ijkm1 = -1.0f;
            }
            if (flags.isFluid(i, j, k + 1, b)) {
              ijkp1 = -1.0f;
            }
          }

          // Set the matrix values now. Setting values in increasing index
          // order as it is done this way by the denseToCSR.
          // Also every example I have seen does it this way.
          if (ijkm1 != 0.0f) {
            // We can't just use the flat index (x + (y * w) + (z * w * h))
            // as the column index because we're operating on a reduced system.
            // Therefore we need to look up the system_index.
            const int isys = THIntTensor_get3d(system_indices, k - 1, j, i);
            if (isys < 0) {
              THError("system index is not defined!");
            }
            THFloatTensor_set1d(val, val_index, ijkm1);
            THIntTensor_set1d(col, val_index, isys);
            val_index++; // increment the val and col place
          }
          if (ijm1k != 0.0f) {
            const int isys = THIntTensor_get3d(system_indices, k, j - 1, i);
            if (isys < 0) {
              THError("system index is not defined!");
            }
            THFloatTensor_set1d(val, val_index, ijm1k);
            THIntTensor_set1d(col, val_index, isys);
            val_index++;
          }
          if (im1jk != 0.0f) {
            const int isys = THIntTensor_get3d(system_indices, k, j, i - 1);
            if (isys < 0) {
              THError("system index is not defined!");
            }
            THFloatTensor_set1d(val, val_index, im1jk);
            THIntTensor_set1d(col, val_index, isys);
            val_index++;
          }

          {  // For scoping of isys.
            const int isys = THIntTensor_get3d(system_indices, k, j, i);
            if (isys < 0) {
              THError("system index is not defined!");
            }
            THFloatTensor_set1d(val, val_index, val_diagonal);
            THIntTensor_set1d(col, val_index, isys);
            val_index++;
          }

          if (ip1jk != 0.0f) {
            const int isys = THIntTensor_get3d(system_indices, k, j, i + 1);
            if (isys < 0) {
              THError("system index is not defined!");
            }
            THFloatTensor_set1d(val, val_index, ip1jk);
            THIntTensor_set1d(col, val_index, isys);
            val_index++;
          }
          if (ijp1k != 0.0f) {
            const int isys = THIntTensor_get3d(system_indices, k, j + 1, i);
            if (isys < 0) {
              THError("system index is not defined!");
            }
            THFloatTensor_set1d(val, val_index, ijp1k);
            THIntTensor_set1d(col, val_index, isys);
            val_index++;
          }
          if (ijkp1 != 0.0f) {
            const int isys = THIntTensor_get3d(system_indices, k + 1, j, i);
            if (isys < 0) {
              THError("system index is not defined!");
            }
            THFloatTensor_set1d(val, val_index, ijkp1);
            THIntTensor_set1d(col, val_index, isys);
            val_index++;
          }

          current_row++;
          THIntTensor_set1d(row, current_row, val_index); 
        } else {  // isFluid & inBounds
          // We shouldn't have got here. All cells in a component should be
          // fluid cells.
          std::cout << "Non fluid cell found in a connected component or "
                    << "fluid cell found on the domain border:"
                    << "  flags(i, j, k, b) = "
                    << flags(i, j, k, b) << std::endl;
          // TODO(tompson): Manta always has 1 solid component on the border,
          // but should we allow it?
          THError("Non fluid cell found in a connected component");
        }
      }
    }
  }

  return val_index;  // Return number of non-zero entries in the matrix A.
}

// allocTempTensor expects a lua table on the stack in index 1, that we will
// store a bunch of temporary tensors. We will allocate these on demand, i.e.
// we will return the existing tensors if they exist, else we will create a new
// one.
template <typename TensorType>
TensorType* allocTempTensor(lua_State* L, const char* name, const char* typeStr,
                            TensorType* (*newFunc)()) {
  TensorType* tensor = nullptr;
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, name);
  // Stack now has:
  // 2, 3, 4, ...: Rest of args to c function.
  // 1: tfluids._tmpPCG
  // -1: tfluids._tmpPCG[name]
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);  // Pop the nil.
    // Create a new tensor.
    tensor = newFunc();
    // Push the new tensor into the table.
    lua_pushstring(L, name);
    luaT_pushudata(L, (void *)tensor, typeStr);
    lua_settable(L, 1);  // Note: pops both key and value.
  } else {
    // Get the value.
    tensor = reinterpret_cast<TensorType*>(luaT_checkudata(L, -1, typeStr));
    // Pop the tensor from the stack.
    lua_pop(L, 1);
  }
  return tensor;
} 

// allocTempCudaTensor is the same as above, except annoyingly the 'new' func
// signature is differnt and there's no easy way to template the function ptr
// without wrapping it with a static number of arguments. It's easier just
// to define two methods, even though it's replicated code.
template <typename TensorType> 
TensorType* allocTempCudaTensor(lua_State* L, const char* name,
                                const char* typeStr,
                                TensorType* (*newFunc)(THCState*)) {
  TensorType* tensor = nullptr;
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, name);
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    tensor = newFunc(cutorch_getstate(L));
    lua_pushstring(L, name);
    luaT_pushudata(L, (void *)tensor, typeStr);
    lua_settable(L, 1);
  } else {
    tensor = reinterpret_cast<TensorType*>(luaT_checkudata(L, -1, typeStr));
    lua_pop(L, 1);
  }
  return tensor;
}

// calpToEpsilon clamps to positive or negative epsilon depending on sign..
inline float clampToEpsilon(const float val, const float epsilon) {
  if (std::abs(val) < epsilon) {
    if (val < 0) {
      return std::min(val, -epsilon);
    } else {
      return std::max(val, epsilon);
    }
  } else {
    return val;
  }
}

__global__ void copyPressureFromSystem(
    THCDeviceTensor<int, 3> system_indices, 
    THCDeviceTensor<float, 1> pressure_pcg, CudaRealGrid pressure,
    const int32_t bout, const float mean) {
  const int32_t xsize = system_indices.getSize(2);
  const int32_t ysize = system_indices.getSize(1);
  const int32_t zsize = system_indices.getSize(0);
  int32_t b, chan, k, j, i;
  // b and chan will always be zero (because we call this on the non-batched
  // tensor).
  if (GetKernelIndices(1, 1, zsize, ysize, xsize, b, chan, k, j, i)) {
    return;
  }
 
  // Look up the system index for the current voxel / pixel.
  int ind = system_indices[k][j][i];
  if (ind < 0) {
    // This pixel wasn't in the linear system (it's a non-fluid cell).
    // The output pressure will be set to zero (but not here since we don't
    // want to overwrite a cell not on our connected component.
  } else {
    pressure(i, j, k, bout) = pressure_pcg[ind] - mean;
  }
}

__global__ void copyDivergenceToSystem(
    THCDeviceTensor<int, 3> system_indices,
    THCDeviceTensor<float, 1> div_pcg, CudaRealGrid div, const int32_t ibatch) {
  const int32_t xsize = system_indices.getSize(2);
  const int32_t ysize = system_indices.getSize(1);
  const int32_t zsize = system_indices.getSize(0);
  int32_t b, chan, k, j, i;
  // b and chan will always be zero (because we call this on the non-batched
  // tensor).
  if (GetKernelIndices(1, 1, zsize, ysize, xsize, b, chan, k, j, i)) {
    return;
  }

  // Look up the system index for the current voxel / pixel.
  const int ind = system_indices[k][j][i];
  if (ind >= 0) {
    // Fluid cell (so it's in the system), copy the divergence.
    div_pcg[ind] = div(i, j, k, ibatch);
  }
}

typedef enum {
  PRECOND_NONE,
  PRECOND_ILU0,
  PRECOND_IC0,
} PrecondType;

PrecondType StringToPrecondType(lua_State* L,
                                const std::string& precond_type_str) {
  if (precond_type_str == "none") {
    return PRECOND_NONE;
  } else if (precond_type_str == "ilu0") {
    return PRECOND_ILU0;
  } else if (precond_type_str == "ic0") {
    return PRECOND_IC0;
  } else {
    luaL_error(L, "precondType is not supported.");
    return PRECOND_NONE;
  }
}

std::string PrecondTypeToString(const PrecondType precond_type) {
  switch (precond_type) {
  case PRECOND_NONE:
    return "none";
  case PRECOND_ILU0:
    return "ilu0";
  case PRECOND_IC0:
    return "ic0";
  default:
    THError("Incorrect precond enum type.");
    exit(-1);
  }
}

static int tfluids_CudaMain_solveLinearSystemPCG(lua_State* L) {
  init_cublas();  // No op if already initialized.
  init_cusparse();  // No op if already initialized.

  THCState* state = cutorch_getstate(L);

  luaL_checktype(L, 1, LUA_TTABLE);  // The first argument should be a table.
  THCudaTensor* p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* flags_gpu = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* div = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 5));
  const std::string precond_type_str =
      static_cast<std::string>(lua_tostring(L, 6));
  const float tol = static_cast<float>(lua_tonumber(L, 7));
  const int64_t max_iter = static_cast<int64_t>(lua_tointeger(L, 8));
  const bool verbose = static_cast<bool>(lua_toboolean(L, 9));

  // This PCG routine uses a LOT of temporary storage. We will create tensors
  // in the tfluids._tmpPCG namespace (table) that are static to the library.
  // This means the tensors stick around between subsequent calls and are
  // resized on demand.

  // The remaining are temporary storage allocated on the lua side but static
  // to the library. They stick around but are resized on demand.
  THIntTensor* system_indices_cpu = allocTempTensor<THIntTensor>(
      L, "systemIndicesCPU", "torch.IntTensor", &THIntTensor_new);
  THIntTensor* row_cpu = allocTempTensor<THIntTensor>(
      L, "rowCPU", "torch.IntTensor", &THIntTensor_new);
  THIntTensor* components = allocTempTensor<THIntTensor>(
      L, "componentsCPU", "torch.IntTensor", &THIntTensor_new);
  THIntTensor* col_cpu = allocTempTensor<THIntTensor>(
      L, "colCPU", "torch.IntTensor", &THIntTensor_new);
  THFloatTensor* val_cpu = allocTempTensor<THFloatTensor>(
      L, "valCPU", "torch.FloatTensor", &THFloatTensor_new);
  THFloatTensor* flags_cpu = allocTempTensor<THFloatTensor>(
      L, "flagsCPU", "torch.FloatTensor", &THFloatTensor_new);

  THCudaIntTensor* row_gpu = allocTempCudaTensor<THCudaIntTensor>(  
      L, "rowGPU", "torch.CudaIntTensor", &THCudaIntTensor_new);
  THCudaIntTensor* col_gpu = allocTempCudaTensor<THCudaIntTensor>(
      L, "colGPU", "torch.CudaIntTensor", &THCudaIntTensor_new);
  THCudaTensor* val_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "valGPU", "torch.CudaTensor", &THCudaTensor_new);

  // TODO(tompson): I'm not convinced we need half of these.
  THCudaTensor* rhs_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "rhsGPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* r_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "rGPU", "torch.CudaTensor", &THCudaTensor_new);  // residual vector
  THCudaTensor* val_precond_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "valILU0GPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* x_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "xGPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* d_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "dGPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* y_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "yGPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* zm1_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "zm1GPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* zm2_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "zm2GPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* p_gpu = allocTempCudaTensor<THCudaTensor>(  // Search direction.
      L, "pGPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* omega_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "omegaGPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaTensor* rm2_gpu = allocTempCudaTensor<THCudaTensor>(
      L, "rm2GPU", "torch.CudaTensor", &THCudaTensor_new);
  THCudaIntTensor* system_indices_gpu = allocTempCudaTensor<THCudaIntTensor>(
      L, "systemIndicesGPU", "torch.CudaIntTensor", &THCudaIntTensor_new);

  // We need the FLAG grid on the CPU, because that's where we're going to
  // construct the sparse matrix (Laplacian).
  THFloatTensor_resize5d(flags_cpu, flags_gpu->size[0], flags_gpu->size[1],
                         flags_gpu->size[2], flags_gpu->size[3],
                         flags_gpu->size[4]);
  THFloatTensor_copyCuda(state, flags_cpu, flags_gpu);  // flags_cpu = flags_gpu
  tfluids_FloatFlagGrid flags(flags_cpu, is_3d);

  CudaRealGrid pressure = toCudaRealGrid(state, p, is_3d);
  CudaRealGrid divergence = toCudaRealGrid(state, div, is_3d);

  // Zero the pressure everywhere, this will zero out the pressure of the
  // non-fluid and empty region cells (because we don't touch them during the
  // pressure solve).
  THCudaTensor_zero(state, p);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  // We wont parallelize over batches, but process each sequentially.
  // TODO(tompson): We could at least parallelize all the Laplacian setups
  // over batch.
  float max_residual = -std::numeric_limits<float>::infinity();
  for (int32_t ibatch = 0; ibatch < nbatch; ibatch++) {
    // Find connected components of fluid regions. If we combine these into a
    // single system then it will be singular (with non-positive pivot) and ICU0
    // preconditioner will fail. Bridson talks about enforcing compatibility
    // conditioner by adding the null-space components to the RHS, this is one
    // solution. Another solution is to solve M PCG problems for each of the M
    // components (this is what we'll do).
    THIntTensor_resize3d(components, zsize, ysize, xsize);
    std::vector<int32_t> component_sizes;
    std::vector<Int3> single_components;
    const int32_t ncomponents = findConnectedFluidComponents(
       flags, components, ibatch, &component_sizes);

    // Now solve ncomponents linear systems.
    for (int32_t icomponent = 0; icomponent < ncomponents; icomponent++) {
      PrecondType precond_type = StringToPrecondType(L, precond_type_str);
      if (component_sizes[icomponent] == 1) {
        // Single components will not have a valid solution. Leave the pressure
        // at zero.
        if (verbose) {
          std::cout << "PCG batch " << (ibatch + 1) << " component "
                    << (icomponent + 1) << " has size 1, skipping."
                    << std::endl;
        }
        continue;
      } else {
        if (verbose) {
          std::cout << "PCG batch " << (ibatch + 1) << " component "
                    << (icomponent + 1) << " has size "
                    << component_sizes[icomponent] << "." << std::endl;
        }
        if (component_sizes[icomponent] < 5) {
          // Don't use a preconditioner, it's slower.
          precond_type = PRECOND_NONE;
        }
      }
      if (verbose) {
        std::cout << "PCG: " << (ibatch + 1) << " component "
                  << (icomponent + 1) << " using precond type "
                  << PrecondTypeToString(precond_type) << std::endl;
      }

      // We're going to create the sparse laplacian next, but we don't want all
      // zero rows (caused by obstacle cells). It guarantees that A is singular.
      // it causes issues with the preconditioner in cusparse, and it is
      // inefficient. Therefore we need to scan through the dataset and create
      // a map of fluid cells, with indices into our new system.
      THIntTensor_resize3d(system_indices_cpu, zsize, ysize, xsize);
      THCudaIntTensor_resize3d(state, system_indices_gpu, zsize, ysize, xsize);
      const int64_t numel = createReducedSystemIndices(
          flags, components, system_indices_cpu, ibatch, icomponent);
  
      // While we're at it, copy these system indices to the GPU (we'll need
      // them later).
      THCudaIntTensor_copyInt(state, system_indices_gpu, system_indices_cpu);
  
      // Recall: resize ops are a no-op if the storage shrinks or stays the
      // same.
      // Note: here we'll allocate the col and val arrays to the maximum
      // possible size (6 neighbors + 1 diagonal = 7). This would be for a
      // domain of all fluid cells, where we also include the border (which like
      // Manta we do not) and if the border cells had neighbors outside (which
      // they do not). So actually this is a conservative sizing.
      // If this is a problem we can always do two passes, one to get the number
      // of non-zero values, and one to fill them (as @kristofe used to do).
      THIntTensor_resize1d(row_cpu, numel + 1);
      THIntTensor_resize1d(col_cpu, numel * 7);
      THFloatTensor_resize1d(val_cpu, numel * 7);
     
      const bool upper_tri = precond_type == PRECOND_IC0;
      const int64_t nz = setupLaplacian(flags, ibatch, row_cpu, col_cpu,
                                        val_cpu, upper_tri, system_indices_cpu,
                                        components, icomponent);
      if (nz > col_cpu->size[0]) {
        luaL_error(L,
                   "INTERNAL ERROR: num of non-zero elements is too large!.");
      }
     
      // Copy the sparse matrix values to the GPU, this time we'll only allocate
      // the number of non-zero values needed.
      THCudaIntTensor_resize1d(state, row_gpu, numel + 1);
      THCudaIntTensor_copyInt(state, row_gpu, row_cpu);
      THCudaIntTensor_resize1d(state, col_gpu, nz);
      {  // Wrap for scoping of col_cpu_nz.
        // Recall: newNarrow(tensor, dim, first_index, size).
        THIntTensor* col_cpu_nz = THIntTensor_newNarrow(col_cpu, 0, 0, nz);
        THCudaIntTensor_copyInt(state, col_gpu, col_cpu_nz);
        THIntTensor_free(col_cpu_nz);
      }
      THCudaTensor_resize1d(state, val_gpu, nz);
      {  // Wrap for scoping of val_cpu_nz.
        THFloatTensor* val_cpu_nz = THFloatTensor_newNarrow(val_cpu, 0, 0, nz);
        THCudaTensor_copyFloat(state, val_gpu, val_cpu_nz);
        THFloatTensor_free(val_cpu_nz);
      }
      
      // Create a description in cusparse of the A matrix that we've
      // created (the val, row and col values above).
      cusparseMatDescr_t descr = 0;
      CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
      if (precond_type == PRECOND_IC0) {
        CHECK_CUSPARSE(cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER));
        CHECK_CUSPARSE(cusparseSetMatType(descr,
                                          CUSPARSE_MATRIX_TYPE_SYMMETRIC));
      } else {
        CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
      }
      CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));   
      // Also copy the rhs (aka div) to the 'b' tensor with reduced indices
      // and at the current batch index.
      THCudaTensor_resize1d(state, rhs_gpu, numel);
      THCDeviceTensor<int, 3> dev_inds =
          toDeviceTensor<int, 3>(state, system_indices_gpu);
      THCDeviceTensor<float, 1> dev_rhs =
          toDeviceTensor<float, 1>(state, rhs_gpu);
      LaunchKernel(L, &copyDivergenceToSystem, 1, 1, zsize, ysize, xsize,
                   dev_inds, dev_rhs, divergence, ibatch);
     
      // Generate the Preconditioner.
      // Create the analysis info object for the A matrix.
      cusparseSolveAnalysisInfo_t info_a = 0;
      cusparseSolveAnalysisInfo_t info_u = 0;
      cusparseSolveAnalysisInfo_t info_ut = 0;  // Only used by ic0.
      cusparseMatDescr_t descr_l = 0;
      cusparseMatDescr_t descr_u = 0;
      if (precond_type != PRECOND_NONE) {
        THCudaTensor_resize1d(state, val_precond_gpu, nz);
        THCudaTensor_copy(state, val_precond_gpu, val_gpu);

        CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_a));
        CHECK_CUSPARSE(cusparseScsrsv_analysis(
          cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, nz,
          descr, DEV_PTR(val_gpu), DEV_INT_PTR(row_gpu), DEV_INT_PTR(col_gpu),
          info_a));

        if (precond_type == PRECOND_ILU0) {
          if (verbose) {
            std::cout << "PCG: Generating ILU0 preconditioner." << std::endl;
          }
          // Generate the Incomplete LU factor H for the matrix A.
          CHECK_CUSPARSE(cusparseScsrilu0(
              cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, descr,
              DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_a));
  
          // Create info objects for the ILU0 preconditioner.
          CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_u));
  
          CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_l));
          CHECK_CUSPARSE(cusparseSetMatType(
              descr_l, CUSPARSE_MATRIX_TYPE_GENERAL));
          CHECK_CUSPARSE(cusparseSetMatIndexBase(
              descr_l, CUSPARSE_INDEX_BASE_ZERO));
          CHECK_CUSPARSE(cusparseSetMatFillMode(
              descr_l, CUSPARSE_FILL_MODE_LOWER));
          CHECK_CUSPARSE(cusparseSetMatDiagType(
              descr_l, CUSPARSE_DIAG_TYPE_UNIT));
  
          CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_u));
          CHECK_CUSPARSE(cusparseSetMatType(
              descr_u, CUSPARSE_MATRIX_TYPE_GENERAL));
          CHECK_CUSPARSE(cusparseSetMatIndexBase(
              descr_u, CUSPARSE_INDEX_BASE_ZERO));
          CHECK_CUSPARSE(cusparseSetMatFillMode(
              descr_u, CUSPARSE_FILL_MODE_UPPER));
          CHECK_CUSPARSE(cusparseSetMatDiagType(
              descr_u, CUSPARSE_DIAG_TYPE_NON_UNIT));
  
          CHECK_CUSPARSE(cusparseScsrsv_analysis(
              cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, nz,
              descr_u, DEV_PTR(val_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_u));
        } else if (precond_type == PRECOND_IC0) {
          if (verbose) {
            std::cout << "PCG: Generating IC0 preconditioner." << std::endl;
          }
          CHECK_CUSPARSE(cusparseScsric0(
              cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, descr,
              DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_a));
 
          CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_u));
          CHECK_CUSPARSE(cusparseSetMatType(
              descr_u, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
          CHECK_CUSPARSE(cusparseSetMatIndexBase(
              descr_u, CUSPARSE_INDEX_BASE_ZERO));
          CHECK_CUSPARSE(cusparseSetMatFillMode(
              descr_u, CUSPARSE_FILL_MODE_UPPER));

          CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_ut));
          CHECK_CUSPARSE(cusparseScsrsv_analysis(
              cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, numel, nz,
              descr_u, DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_ut));

          CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_u));
          CHECK_CUSPARSE(cusparseScsrsv_analysis(
              cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, nz,
              descr_u, DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_u));
        } else {
          luaL_error(L, "Incorrect preconType ('none', 'ic0', 'ilu0')");
        }
      }
  
      // While we're at it, set the pressure value to a zero value (note we
      // could also use the previous frame's pressure). This is CG's initial
      // guess.
      THCudaTensor_resize1d(state, x_gpu, numel);
      THCudaTensor_zero(state, x_gpu);
  
      // TODO(tompson): Move all these to the start of the function.
      THCudaTensor_resize1d(state, y_gpu, numel);
      THCudaTensor_resize1d(state, p_gpu, numel);
      THCudaTensor_resize1d(state, omega_gpu, numel);
      THCudaTensor_resize1d(state, zm1_gpu, numel);
      THCudaTensor_resize1d(state, zm2_gpu, numel);
      THCudaTensor_resize1d(state, rm2_gpu, numel);
      // TODO(tompson): Do we need yet another copy of the RHS?
     
      // The algorithm we're implementing here is from Matrix Computations,
      // Golub and Van Loan, Algorithm 10.3.1:
   
      int32_t iter = 0;
      float r_norm_sq1;  // r_norm_sq1 is the current residual
      float r_norm_sq0;  // r_norm_sq0 is the previous iteration's residual
      // Since we start with x = 0, the initial residual is just the norm of the
      // rhs (i.e. residual = ||rhs - A * x|| = ||rhs||)
      THCudaTensor_resize1d(state, r_gpu, numel);  // residual
      THCudaTensor_copy(state, r_gpu, rhs_gpu);
      CHECK_CUBLAS(cublasSdot(cublas_handle, numel, DEV_PTR(r_gpu), 1,
                              DEV_PTR(r_gpu), 1, &r_norm_sq1));

      if (isnan(r_norm_sq1)) {
        luaL_error(L, "PCG Error: starting residual is nan!");
      }

      const float one = 1.0f;
      const float zero = 0.0f;
      float numerator;
      float denominator;
      float beta;
      float alpha;
      float nalpha;
      if (verbose) {
        std::cout << "PCG batch " << (ibatch + 1) << " comp "
                  << (icomponent + 1) << ": starting residual "
                  << std::sqrt(r_norm_sq1) << " (tol " << tol
                  << ", precondType = " << PrecondTypeToString(precond_type)
                  << ")" << std::endl;
      }
  
      // epsilon ~= 1e-38 (just prevents divide by zero).
      const float epsilon = std::numeric_limits<float>::min();
      while (r_norm_sq1 > tol * tol && iter <= max_iter) {
        if (precond_type == PRECOND_ILU0) {
          // Solve M * z_k = r_k
          CHECK_CUSPARSE(cusparseScsrsv_solve(
              cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, &one,
              descr_l, DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_a, DEV_PTR(r_gpu), DEV_PTR(y_gpu)));
          CHECK_CUSPARSE(cusparseScsrsv_solve(
              cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, &one,
              descr_u, DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_u, DEV_PTR(y_gpu),
              DEV_PTR(zm1_gpu)));
        } else if (precond_type == PRECOND_IC0) {
          CHECK_CUSPARSE(cusparseScsrsv_solve(
              cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, numel, &one,
              descr_u, DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_ut, DEV_PTR(r_gpu), DEV_PTR(y_gpu)));
          CHECK_CUSPARSE(cusparseScsrsv_solve(
              cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, &one,
              descr_u, DEV_PTR(val_precond_gpu), DEV_INT_PTR(row_gpu),
              DEV_INT_PTR(col_gpu), info_u, DEV_PTR(y_gpu),
              DEV_PTR(zm1_gpu)));
        }
  
        iter++;  // k = k + 1
  
        // Calculate the next search direction p_k.
        if (iter == 1) {
          if (precond_type != PRECOND_NONE) {
            THCudaTensor_copy(state, p_gpu, zm1_gpu);  // p_1 = z_0
          } else {
            THCudaTensor_copy(state, p_gpu, r_gpu);  // p_1 = r_0
          }
        } else {
          if (precond_type != PRECOND_NONE) {
            // beta_k = r_{k_1}^T * z_{k - 1} / (r_{k-2}^T * z_{k - 2})
            CHECK_CUBLAS(cublasSdot(cublas_handle, numel, DEV_PTR(r_gpu), 1,
                                    DEV_PTR(zm1_gpu), 1, &numerator));
            CHECK_CUBLAS(cublasSdot(cublas_handle, numel, DEV_PTR(rm2_gpu), 1,
                                    DEV_PTR(zm2_gpu), 1, &denominator));
            beta = numerator / clampToEpsilon(denominator, epsilon);
            // p_k = z_{k - 1} + beta_k * p_{k - 1}
            CHECK_CUBLAS(cublasSscal(
                cublas_handle, numel, &beta, DEV_PTR(p_gpu), 1));
            CHECK_CUBLAS(cublasSaxpy(
                cublas_handle, numel, &one, DEV_PTR(zm1_gpu), 1,
                DEV_PTR(p_gpu), 1));
  
          } else {
            beta = r_norm_sq1 / clampToEpsilon(r_norm_sq0, epsilon);
            CHECK_CUBLAS(cublasSscal(cublas_handle, numel, &beta,
                                     DEV_PTR(p_gpu), 1));
            CHECK_CUBLAS(cublasSaxpy(cublas_handle, numel, &one,
                                     DEV_PTR(r_gpu), 1, DEV_PTR(p_gpu), 1));
          }
        }
  
        // alpha_k = r_{k-1}^T * z_{k - 1} / (p_k^T * A * p_k)
        // omega_k = A * p_k.
        // Recall: cusparseScsrmv is a sparse matrix-vec + const operator.
        // TODO(tompson): should the sparse descr be descr_u?
        CHECK_CUSPARSE(cusparseScsrmv(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numel, numel,
            nz, &one, descr, DEV_PTR(val_gpu), DEV_INT_PTR(row_gpu),
            DEV_INT_PTR(col_gpu), DEV_PTR(p_gpu), &zero,
            DEV_PTR(omega_gpu)));
        if (precond_type != PRECOND_NONE) {
          // numerator = r_{k-1}^T * z_{k - 1}
          CHECK_CUBLAS(cublasSdot(cublas_handle, numel, DEV_PTR(r_gpu), 1,
                                  DEV_PTR(zm1_gpu), 1, &numerator));
          // denominator = p_k^T * A * p_k = p_k^T * omega_k
          CHECK_CUBLAS(cublasSdot(cublas_handle, numel, DEV_PTR(p_gpu), 1,
                                  DEV_PTR(omega_gpu), 1, &denominator));
        } else {
          numerator = r_norm_sq1;
          CHECK_CUBLAS(cublasSdot(cublas_handle, numel, DEV_PTR(p_gpu), 1,
                                  DEV_PTR(omega_gpu), 1, &denominator));
        }
        alpha = numerator / clampToEpsilon(denominator, epsilon);
        
        // x_k = x_{k - 1} + alpha_k * p_k
        // Recall: cublasSaxpy(handle, n, alpha, x, incx, y, incy) performs:
        // --> y [ j ] = alpha × x [ k ] + y [ j ]
        CHECK_CUBLAS(cublasSaxpy(
            cublas_handle, numel, &alpha, DEV_PTR(p_gpu), 1, DEV_PTR(x_gpu),
            1));
        if (precond_type != PRECOND_NONE) {
          THCudaTensor_copy(state, rm2_gpu, r_gpu);  // rm2_gpu = r_gpu
          THCudaTensor_copy(state, zm2_gpu, zm1_gpu);  // zm2_gpu = zm1_gpu
        }
        nalpha = -alpha;
        // According to Shewchuck we should re-init r every 50 iterations.
        // EDIT(tompson): It doesn't seem to help (removed but in git history).

        // r_k = r_{k - 1} - alpha_k * A * p_k = r_{k - 1} - alpha_k * omega_k
        CHECK_CUBLAS(cublasSaxpy(cublas_handle, numel, &nalpha,
                                 DEV_PTR(omega_gpu), 1, DEV_PTR(r_gpu), 1));
  
        r_norm_sq0 = r_norm_sq1;  // Update previous residual.
  
        // Finally, calculate the new residual.
        CHECK_CUBLAS(cublasSdot(cublas_handle, numel, DEV_PTR(r_gpu), 1,
                                DEV_PTR(r_gpu), 1, &r_norm_sq1));
  
        if (verbose) {
			
          std::cout << "PCG batch " << (ibatch + 1) << " comp " 
                    << (icomponent + 1) << " iter " << iter << ": residual "
                    << std::sqrt(r_norm_sq1) << " (tol " << tol
                    << ", precondType = " << PrecondTypeToString(precond_type)
                    << ")" << std::endl;
        }
  
        if (isnan(r_norm_sq1)) {
          luaL_error(L, "ERROR: r_norm_sq1 is nan!");
        }
      }
      
      if (verbose) {
        if (iter == max_iter) {
          std::cout << "PCG batch " << (ibatch + 1) << " component "
                    << (icomponent + 1) << " hit max iteration count ("
                    << max_iter << ")" << std::endl;
        } else if (r_norm_sq1 < tol * tol) {
          std::cout << "PCG batch " << (ibatch + 1) << " component "
                    << (icomponent + 1) << " residual " << std::sqrt(r_norm_sq1)
                    << " fell below tol (" << tol << ")" << std::endl;
        }
      }
  
      max_residual = std::max(max_residual, std::sqrt(r_norm_sq1));
 
      // For each separate linear system we're free to choose whatever constant
      // DC term we want. This has no impact on the velocity update, but will
      // be important if we include a pressure term when training the convnet
      // (which doesn't like arbitrary output scales / offsets).
      const float x_mean = THCudaTensor_meanall(state, x_gpu);     

      // The result is in x_gpu. However, this is the pressure in the reduced
      // system (with non-fluid cells removed), we need to copy this back to the
      // original Cartesian (d x w x h) system.
      THCDeviceTensor<float, 1> dev_x = toDeviceTensor<float, 1>(state, x_gpu);
      LaunchKernel(L, &copyPressureFromSystem, 1, 1, zsize, ysize, xsize,
                   dev_inds, dev_x, pressure, ibatch, x_mean);
  
      // Clean up cusparse.
      // TODO(tompson): Is there anything else to do?
      CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
      if (precond_type != PRECOND_NONE) {
        CHECK_CUSPARSE(cusparseDestroySolveAnalysisInfo(info_a));
        CHECK_CUSPARSE(cusparseDestroySolveAnalysisInfo(info_u));
        CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_l));
        CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_u));
        if (precond_type == PRECOND_IC0) {
          CHECK_CUSPARSE(cusparseDestroySolveAnalysisInfo(info_ut));
        }
      }
    } // for each connected component
  }  // for each batch.

  lua_pushnumber(L, max_residual);
  return 1;
}

//******************************************************************************
// solveLinearSystemJacobi
//******************************************************************************

__global__ void kernel_jacobiIteration(
    CudaFlagGrid flags, CudaRealGrid div,
    CudaRealGrid pressure, CudaRealGrid prev_pressure, const int bnd) {
  int32_t b, chan, k, j, i;
  if (GetKernelIndices(flags, b, chan, k, j, i)) {
    return;
  }

  if (i < bnd || i > flags.xsize() - 1 - bnd ||
      j < bnd || j > flags.ysize() - 1 - bnd ||
      (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
    pressure(i, j, k, b) = 0;  // Zero pressure on the border.
    return;
  }
  
  if (flags.isObstacle(i, j, k, b)) {
    pressure(i, j, k, b) = 0;
    return;
  }

  // Otherwise in a fluid or empty cell.
  // TODO(tompson): Is the logic here correct? Should empty cells be non-zero?
  const float divergence = div(i, j, k, b);
  
  // Get all the neighbors
  const float pC = prev_pressure(i, j, k, b);

  float p1 = prev_pressure(i - 1, j, k, b);
  float p2 = prev_pressure(i + 1, j, k, b);
  float p3 = prev_pressure(i, j - 1, k, b);
  float p4 = prev_pressure(i, j + 1, k, b);
  float p5 = flags.is_3d() ? prev_pressure(i, j, k - 1, b) : 0;
  float p6 = flags.is_3d() ? prev_pressure(i, j, k + 1, b) : 0;

  if (flags.isObstacle(i - 1, j, k, b)) {
    p1 = pC;
  }
  if (flags.isObstacle(i + 1, j, k, b)) {
    p2 = pC;
  }
  if (flags.isObstacle(i, j - 1, k, b)) {
    p3 = pC;
  }
  if (flags.isObstacle(i, j + 1, k, b)) {
    p4 = pC;
  }
  if (flags.is_3d() && flags.isObstacle(i, j, k - 1, b)) {
    p5 = pC;
  }
  if (flags.is_3d() && flags.isObstacle(i, j, k + 1, b)) {
    p6 = pC;
  }

  const float denom = flags.is_3d() ? 6.0f : 4.0f;
  const float v = (p1 + p2 + p3 + p4 + p5 + p6 + divergence) / denom;
  pressure(i, j, k, b) = v;
}

static int tfluids_CudaMain_solveLinearSystemJacobi(lua_State* L) {
  THCState* state = cutorch_getstate(L);

  THCudaTensor* tensor_p = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 1, "torch.CudaTensor"));
  THCudaTensor* tensor_flags = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 2, "torch.CudaTensor"));
  THCudaTensor* tensor_div = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 3, "torch.CudaTensor"));
  THCudaTensor* tensor_p_prev = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 4, "torch.CudaTensor"));
  THCudaTensor* tensor_p_delta = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 5, "torch.CudaTensor"));
  THCudaTensor* tensor_p_delta_norm = reinterpret_cast<THCudaTensor*>(
      luaT_checkudata(L, 6, "torch.CudaTensor"));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 7));
  const float p_tol = static_cast<float>(lua_tonumber(L, 8));
  const int64_t max_iter = static_cast<int64_t>(lua_tointeger(L, 9));
  const bool verbose = static_cast<int64_t>(lua_toboolean(L, 10));

  if (max_iter < 1) {
    luaL_error(L, "At least 1 iteration is needed (maxIter < 1)");
    return 0;
  }

  CudaFlagGrid flags = toCudaFlagGrid(state, tensor_flags, is_3d);
  CudaRealGrid pressure = toCudaRealGrid(state, tensor_p, is_3d);
  CudaRealGrid pressure_prev = toCudaRealGrid(state, tensor_p_prev, is_3d);
  CudaRealGrid div = toCudaRealGrid(state, tensor_div, is_3d);

  // Initialize the pressure to zero.
  THCudaTensor_zero(state, tensor_p);
  THCudaTensor_zero(state, tensor_p_prev);

  // Start with the output of the next iteration going to pressure.
  CudaRealGrid* cur_pressure = &pressure;
  CudaRealGrid* cur_pressure_prev = &pressure_prev;

  const int32_t nbatch = flags.nbatch();
  const int64_t xsize = flags.xsize();
  const int64_t ysize = flags.ysize();
  const int64_t zsize = flags.zsize();
  const int64_t numel = xsize * ysize * zsize;

  float residual;
  int64_t iter = 0;
  while (true) {
    const int32_t bnd = 1;
    // LaunchKernel args: lua_State, func, domain, args...
    LaunchKernel(L, &kernel_jacobiIteration, flags,
                 flags, div, *cur_pressure, *cur_pressure_prev, bnd);

    // Current iteration output is now in cur_pressure (wherever that points).

    // Calculate the change in pressure up to a sign (i.e. the sign might be
    // incorrect, but we don't care).
    THCudaTensor_csub(state, tensor_p_delta, tensor_p, 1.0f, tensor_p_prev);
    THCudaTensor_resize2d(state, tensor_p_delta, nbatch, numel);
    // Calculate L2 norm over dim 2.
    THCudaTensor_norm(state, tensor_p_delta_norm, tensor_p_delta, 2, 1, 1);
    // Put the view back.
    THCudaTensor_resize5d(state, tensor_p_delta, nbatch, 1, zsize, ysize,
                          xsize);
    residual = THCudaTensor_maxall(state, tensor_p_delta_norm);
    if (verbose) {
      std::cout << "Jacobi iteration " << (iter + 1) << ": residual "
                << residual << std::endl;
    }
    // TODO(tompson) calculate divergence and implement divtol (it'll make it
    // slower).
    // TODO(tompson): We terminate on the worst case batch is this OK?
    if (residual < p_tol) {
      if (verbose) {
        std::cout << "Jacobi max residual fell below p_tol (" << p_tol
                  << ") (terminating)" << std::endl;
      }
      break;
    }

    iter++;
    if (iter >= max_iter) {
      if (verbose) {
        std::cout << "Jacobi max iteration count (" << max_iter
                  << ") reached (terminating)" << std::endl;
      }
      break;
    }

    // We haven't yet terminated.
    CudaRealGrid* tmp = cur_pressure;
    cur_pressure = cur_pressure_prev;
    cur_pressure_prev = tmp;
  }

  // If we terminated with the cur_pressure pointing to the tmp array, then we
  // have to copy the pressure back into the output tensor.
  if (cur_pressure == &pressure_prev) {
    THCudaTensor_copy(state, tensor_p, tensor_p_prev);  // p = p_prev
  }

  // Note, mean-subtraction is performed on the lua side.

  lua_pushnumber(L, residual);
  return 1;
}

//******************************************************************************
// INIT METHODS
//******************************************************************************
static const struct luaL_Reg tfluids_CudaMain__ [] = {
  {"advectScalar", tfluids_CudaMain_advectScalar},
  {"advectVel", tfluids_CudaMain_advectVel},
  {"setWallBcsForward", tfluids_CudaMain_setWallBcsForward},
  {"vorticityConfinement", tfluids_CudaMain_vorticityConfinement},
  {"addBuoyancy", tfluids_CudaMain_addBuoyancy},
  {"addGravity", tfluids_CudaMain_addGravity},
  {"velocityUpdateForward", tfluids_CudaMain_velocityUpdateForward},
  {"velocityUpdateBackward", tfluids_CudaMain_velocityUpdateBackward},
  {"velocityDivergenceForward", tfluids_CudaMain_velocityDivergenceForward},
  {"velocityDivergenceBackward", tfluids_CudaMain_velocityDivergenceBackward},
  {"emptyDomain", tfluids_CudaMain_emptyDomain},
  {"flagsToOccupancy", tfluids_CudaMain_flagsToOccupancy},
  {"solveLinearSystemPCG", tfluids_CudaMain_solveLinearSystemPCG},
  {"solveLinearSystemJacobi", tfluids_CudaMain_solveLinearSystemJacobi},
  {"volumetricUpSamplingNearestForward",
   tfluids_CudaMain_volumetricUpSamplingNearestForward},
  {"volumetricUpSamplingNearestBackward",
   tfluids_CudaMain_volumetricUpSamplingNearestBackward},
  {"signedDistanceField", tfluids_CudaMain_signedDistanceField},
  {NULL, NULL}  // NOLINT
};

const struct luaL_Reg* tfluids_CudaMain_getMethodsTable() {
  return tfluids_CudaMain__;
}

void tfluids_CudaMain_init(lua_State* L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, tfluids_CudaMain__, "tfluids");
}
