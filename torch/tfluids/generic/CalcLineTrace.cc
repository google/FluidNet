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

#include <stdio.h>
#include <string>
#include "stdint.h"
#include <math.h>
#include <memory>
#include "mex.h"

#define MATLAB_BUILD

#define TH_CONCAT_4(x,y,z,w) TH_CONCAT_4_EXPAND(x,y,z,w)
#define TH_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
#define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)
#define TH_CONCAT_3_EXPAND(x,y,z) x ## y ## z
#define THTensor          TH_CONCAT_3(TH,Real,Tensor)
#define THTensor_(NAME)   TH_CONCAT_4(TH,Real,Tensor_,NAME)
#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define tfluids_(NAME) TH_CONCAT_3(tfluids_, Real, NAME)
#define real float
#define accreal double
#define Real Float
#define THInf FLT_MAX
#define TH_REAL_IS_FLOAT
#define __host__
#define __device__

void THError(const char* c_str) {
  mexErrMsgIdAndTxt("MATLAB:CalcLineTrace:error", c_str);
}

// Create our own fake tensor class.
typedef struct THTensor {
    std::unique_ptr<long[]> size;
    std::unique_ptr<long[]> stride;
    int nDimension;
    std::unique_ptr<real[]> storage;
} THTensor;
real* THTensor_(data)(const THTensor* self) {
  return self->storage.get();
}

#include "generic/stack_trace.cc"
#include "generic/int3.cu.h"

inline int32_t clamp(const int32_t x, const int32_t low, const int32_t high) {
  return std::max<int32_t>(std::min<int32_t>(x, high), low);
}

#include "generic/vec3.cc"
#include "third_party/cell_type.h"
#include "third_party/grid.cc"
#include "calc_line_trace.cc"

using namespace std;

#define mexAssert(cond, str) \
{ \
  if (!(cond)) { \
    mexErrMsgIdAndTxt("MATLAB:CalcLineTrace:failedAssertion", str); \
  } \
}

void getVec3(const mxArray* prhs, tfluids_(vec3)* pos) {
  mexAssert(mxGetNumberOfDimensions(prhs) == 2, "ERROR: vector not 2D");
  const mwSize* sz = mxGetDimensions(prhs);
  mexAssert(sz[0] == 1 && sz[1] == 3, "ERROR: vector not size [1, 3]");
  const double* posMat = mxGetPr(prhs);
  pos->x = posMat[0];
  pos->y = posMat[1];
  pos->z = posMat[2];
}

void getInt3(const mxArray* prhs, Int3* pos) {
  mexAssert(mxGetNumberOfDimensions(prhs) == 2, "ERROR: vector not 2D");
  const mwSize* sz = mxGetDimensions(prhs);
  mexAssert(sz[0] == 1 && sz[1] == 3, "ERROR: vector not size [1, 3]");
  const double* posMat = mxGetPr(prhs);
  pos->x = static_cast<int>(posMat[0]);
  pos->y = static_cast<int>(posMat[1]);
  pos->z = static_cast<int>(posMat[2]);
}


// The gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Only 5 inputs allowed
  if (nrhs != 4) {
    mexErrMsgIdAndTxt("MATLAB:CalcLineTrace:invalidNumInputs", 
      "Input must be pos, delta, dims, obs");
  }
  
  // input must be pos, delta, dims, obs.
  for (int i = 0; i < 4; i++) {
    if (mxIsDouble(prhs[i]) != 1) {
      mexErrMsgIdAndTxt("MATLAB:CalcLineTrace:notDouble",
        "Inputs must be double.");
    }
  }
  
  // Check that the inputs are the right size.
  // 0: pos
  tfluids_(vec3) pos;
  getVec3(prhs[0], &pos);
  // mexPrintf("pos = %f %f %f\n", pos.x, pos.y, pos.z);

  // 1: delta
  tfluids_(vec3) delta;
  getVec3(prhs[1], &delta);
  // mexPrintf("delta = %f %f %f\n", delta.x, delta.y, delta.z);
  
  // 2: dims
  Int3 dims;
  getInt3(prhs[2], &dims);
  // mexPrintf("dims = %d %d %d\n", dims.x, dims.y, dims.z);

  // 3: obs
  const int obs_dim = static_cast<int>(mxGetNumberOfDimensions(prhs[3]));
  mexAssert(obs_dim == 3 || obs_dim == 2, "ERROR: obs not 3D or 2D");
  const mwSize* sz = mxGetDimensions(prhs[3]);
  if (obs_dim == 2) {
    mexAssert(sz[0] == dims.x, "ERROR: obs size mismatch.");
    mexAssert(sz[1] == dims.y, "ERROR: obs size mismatch.");
  } else {
    mexAssert(sz[0] == dims.x, "ERROR: obs size mismatch.");
    mexAssert(sz[1] == dims.y, "ERROR: obs size mismatch.");
    mexAssert(sz[2] == dims.z, "ERROR: obs size mismatch.");
  }
  const double* obs = mxGetPr(prhs[3]);

  // Convert the obs to a flag grid.
  THTensor flags_tensor;  // Create a fake tensor.
  flags_tensor.nDimension = 5;
  flags_tensor.size.reset(new long[flags_tensor.nDimension]);
  flags_tensor.stride.reset(new long[flags_tensor.nDimension]);
  flags_tensor.size[0] = 1;  // batch
  flags_tensor.size[1] = 1;  // chan
  flags_tensor.size[2] = dims.z;
  flags_tensor.size[3] = dims.y;
  flags_tensor.size[4] = dims.x;
  flags_tensor.stride[4] = 1;
  flags_tensor.stride[3] = dims.x;
  flags_tensor.stride[2] = dims.x * dims.y;
  flags_tensor.stride[1] = dims.x * dims.y * dims.z;
  flags_tensor.stride[0] = dims.x * dims.y * dims.z * 1;
  flags_tensor.storage.reset(new real[dims.z * dims.y * dims.x]);
  for (int k = 0; k < dims.z; k++) {
    for (int j = 0; j < dims.y; j++) {
      for (int i = 0; i < dims.x; i++) {
        const int index = k * dims.x * dims.y + j * dims.x + i;
        if (obs[index] > 0) {
          flags_tensor.storage[index] = TypeObstacle;
        } else {
          flags_tensor.storage[index] = TypeFluid;
        }
      }
    }
  }

  bool is_3d = dims.z > 1;
  tfluids_(FlagGrid) flags(&flags_tensor, is_3d);

  // Call the actual C function.
  tfluids_(vec3) new_pos;
  const int32_t ibatch = 0;
  const bool do_trace = true;
  const bool collided = calcLineTrace(pos, delta, flags, ibatch, &new_pos,
                                      do_trace);
 
  // Only 2 outputs allowed
  if (nlhs != 2) {
    mexErrMsgIdAndTxt("MATLAB:CalcLineTrace:invalidNumOutputs", 
      "Two outputs required: [collide, new_pos].");
  }  
  
  // Allocate the outputs and set them.
  plhs[0] = mxCreateDoubleScalar(static_cast<double>(collided));
  plhs[1] = mxCreateDoubleMatrix(1, 3, mxREAL);
  double* new_pos_mex = mxGetPr(plhs[1]);
  new_pos_mex[0] = new_pos.x;
  new_pos_mex[1] = new_pos.y;
  new_pos_mex[2] = new_pos.z;
}
