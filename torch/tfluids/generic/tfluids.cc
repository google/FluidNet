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

#include <assert.h>
#include <memory>

// Note: No need to pull in vec.h and grid.h because the init.cu puts
// them inline (and above) these function definitions.

// There are a LOT of methods in tfluids that borrow heavily (or port) parts of
// Manta. These are compiled here but note that they are added under a separate
// license. You should see FluidNet/torch/tfluids/third_party/README for more
// information.
#include "third_party/tfluids.cc"

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

// *****************************************************************************
// velocityDivergenceBackward
// *****************************************************************************

static int tfluids_(Main_velocityDivergenceBackward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 4));
  THTensor* tensor_grad_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(RealGrid) grad_output(tensor_grad_output, is_3d);
  tfluids_(MACGrid) grad_u(tensor_grad_u, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();


  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    // Firstly, we're going to accumulate gradient contributions, so set
    // grad_u to 0.
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) { 
      for (j = 0; j < ysize; j++) { 
        for (i = 0; i < xsize; i++) {
          grad_u.setSafe(i, j, k, b, tfluids_(vec3)(0, 0, 0));
        }
      }
    }

    // Now accumulate gradients from across the output gradient.
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
            // Manta zeros stuff on the border in the forward pass, so they do
            // not contribute gradient.
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            // Blocked cells don't contribute gradient.
            continue;
          }

          // TODO(tompson): Can we restructure this into a gather rather than
          // a scatter? (it would mean multiple redundant lookups into flags,
          // but it might be faster...).
          const real go = grad_output(i, j, k, b);
#pragma omp atomic
          grad_u(i, j, k, 0, b) += go;
#pragma omp atomic
          grad_u(i + 1, j, k, 0, b) -= go;
#pragma omp atomic
          grad_u(i, j, k, 1, b) += go;
#pragma omp atomic
          grad_u(i, j + 1, k, 1, b) -= go;
          if (is_3d) {
#pragma omp atomic
            grad_u(i, j, k, 2, b) += go;
#pragma omp atomic
            grad_u(i, j, k + 1, 2, b) -= go;
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// emptyDomain
// *****************************************************************************

static int tfluids_(Main_emptyDomain)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 2));
  const int32_t bnd = static_cast<int32_t>(lua_tointeger(L, 3));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch  = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            flags(i, j, k, b) = TypeObstacle;
          } else {
            flags(i, j, k, b) = TypeFluid;
          }
        }
      }
    }
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// flagsToOccupancy
// *****************************************************************************

static int tfluids_(Main_flagsToOccupancy)(lua_State *L) {
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_occupancy=
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));

  const int32_t numel = THTensor_(numel)(tensor_flags);
  const real* pflags =
      reinterpret_cast<const real*>(THTensor_(data)(tensor_flags));
  real* pocc = reinterpret_cast<real*>(THTensor_(data)(tensor_occupancy));

  if (!THTensor_(isContiguous)(tensor_flags) ||
      !THTensor_(isContiguous)(tensor_occupancy)) {
    luaL_error(L, "ERROR: tensors are not contiguous!");
  }

  int32_t i;
  bool bad_cell = false;
#pragma omp parallel for private(i)
  for (i = 0; i < numel; i++) {
    const int32_t flag = static_cast<int32_t>(pflags[i]);
    if (flag == TypeFluid) {
      pocc[i] = 0;
    } else if (flag == TypeObstacle) {
      pocc[i] = 1;
    } else {
      bad_cell = true;  // There's no race cond because we'll only trigger once.
    }
  }

  if (bad_cell) {
    luaL_error(L, "ERROR: unsupported flag cell found!");
  }

  return 0;  // Recall: number of return values on the lua stack. 
}

// *****************************************************************************
// velocityUpdateBackward
// *****************************************************************************

static int tfluids_(Main_velocityUpdateBackward)(lua_State *L) {
  // Get the args from the lua stack. NOTE: We do ALL arguments (size checking)
  // on the lua stack.
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* tensor_flags =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* tensor_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* tensor_grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 5));
  THTensor* tensor_grad_p =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 6, torch_Tensor));

  tfluids_(FlagGrid) flags(tensor_flags, is_3d);
  tfluids_(RealGrid) grad_p(tensor_grad_p, is_3d);
  tfluids_(MACGrid) grad_output(tensor_grad_output, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    // Firstly, we're going to accumulate gradient contributions, so set
    // grad_p to 0.
    int32_t k, j, i;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          grad_p(i, j, k, b) = 0;
        }
      }
    }

    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta doesn't touch the velocity on the boundaries (i.e.
            // it stays constant and so has zero gradient).
            continue;
          }

          const tfluids_(vec3) go(grad_output(i, j, k, b));

          if (flags.isFluid(i, j, k, b)) {
            if (flags.isFluid(i - 1, j, k, b)) {
              // fwd: vel(i, j, k, 0, b) -= (p(i, j, k, b) - p(i - 1, j, k, b));
#pragma omp atomic
              grad_p(i, j, k, b) -= go.x;
#pragma omp atomic
              grad_p(i - 1, j, k, b) += go.x;
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              // fwd: vel(i, j, k, 1, b) -= (p(i, j, k, b) - p(i, j - 1, k, b));
#pragma omp atomic
              grad_p(i, j, k, b) -= go.y;
#pragma omp atomic  
              grad_p(i, j - 1, k, b) += go.y;
            }
            if (is_3d && flags.isFluid(i, j, k - 1, b)) {
              // fwd: vel(i, j, k, 2, b) -= (p(i, j, k, b) - p(i, j, k - 1, b));
#pragma omp atomic
              grad_p(i, j, k, b) -= go.z;
#pragma omp atomic  
              grad_p(i, j, k - 1, b) += go.z;
            }

            if (flags.isEmpty(i - 1, j, k, b)) {
              // fwd: vel(i, j, k, 0, b) -= p(i, j, k, b);
#pragma omp atomic
              grad_p(i, j, k, b) -= go.x;
            }
            if (flags.isEmpty(i, j - 1, k, b)) {
              // fwd: vel(i, j, k, 1, b) -= p(i, j, k, b);
#pragma omp atomic
              grad_p(i, j, k, b) -= go.y;
            }
            if (is_3d && flags.isEmpty(i, j, k - 1, b)) {
              // fwd: vel(i, j, k, 2, b) -= p(i, j, k, b);
#pragma omp atomic
              grad_p(i, j, k, b) -= go.z;
            }
          }
          else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
            // don't change velocities in outflow cells   
            if (flags.isFluid(i - 1, j, k, b)) {
              // fwd: vel(i, j, k, 0, b) += p(i - 1, j, k, b);
#pragma omp atomic
              grad_p(i - 1, j, k, b) += go.x;
            } else {
              // fwd: vel(i, j, k, 0, b)  = 0.f;
              // Output doesn't depend on p, so gradient is zero and so doesn't
              // contribute.
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              // fwd: vel(i, j, k, 1, b) += p(i, j - 1, k, b);
#pragma omp atomic
              grad_p(i, j - 1, k, b) += go.y;
            } else {
              // fwd: vel(i, j, k, 1, b)  = 0.f;
              // Output doesn't depend on p, so gradient is zero and so doesn't
              // contribute.
            }
            if (is_3d) {
              if (flags.isFluid(i, j, k - 1, b)) {
                // fwd: vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
#pragma omp atomic
                grad_p(i, j, k - 1, b) += go.z;
              } else {
                // fwd: vel(i, j, k, 2, b)  = 0.f;
                // Output doesn't depend on p, so gradient is zero and so
                // doesn't contribute.
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
// drawVelocityField
// *****************************************************************************

static int tfluids_(Main_drawVelocityField)(lua_State *L) {
#ifdef BUILD_GL_FUNCS
  THTensor* tensor_u =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const bool flip_y = static_cast<bool>(lua_toboolean(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));

  if (tensor_u->nDimension != 5) {
    luaL_error(L, "Input vector field should be 5D.");
  }

  tfluids_(MACGrid) vel(tensor_u, is_3d);

  const int32_t nbatch = vel.nbatch();
  const int32_t xsize = vel.xsize();
  const int32_t ysize = vel.ysize();
  const int32_t zsize = vel.zsize();

  if (nbatch > 1) {
    luaL_error(L, "input velocity field has more than one sample.");
  }

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glBegin(GL_LINES);
  const int32_t bnd = 1;
  for (int32_t b = 0; b < nbatch; b++) {
    for (int32_t z = 0; z < zsize; z++) {
      for (int32_t y = 0; y < ysize; y++) {
        for (int32_t x = 0; x < xsize; x++) {
          if (x < bnd || x > xsize - 1 - bnd ||
              y < bnd || y > ysize - 1 - bnd ||
              (is_3d && (z < bnd || z > zsize - 1 - bnd))) {
            continue;
          }
          tfluids_(vec3) v = vel.getCentered(x, y, z, b);
         
          // Velocity is in grids / second. But we need coordinates in [0, 1].
          v.x = v.x / static_cast<real>(xsize - 1);
          v.y = v.y / static_cast<real>(ysize - 1);
          v.z = is_3d ? v.z / static_cast<real>(zsize - 1) : 0;

          // Same for position.
          real px = static_cast<real>(x) / static_cast<real>(xsize - 1);
          real py = static_cast<real>(y) / static_cast<real>(ysize - 1);
          real pz =
              is_3d ? static_cast<real>(z) / static_cast<real>(zsize - 1) : 0;
          py = flip_y ? py : static_cast<real>(1) - py;
          v.y = flip_y ? -v.y : v.y;
          glColor4f(0.7f, 0.0f, 0.0f, 1.0f);
          glVertex3f(static_cast<float>(px),
                     static_cast<float>(py),
                     static_cast<float>(pz));
          glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
          glVertex3f(static_cast<float>(px + v.x),
                     static_cast<float>(py - v.y),
                     static_cast<float>(pz + v.z));
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

// *****************************************************************************
// loadTensorTexture
// *****************************************************************************

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

// *****************************************************************************
// volumetricUpsamplingNearestForward
// *****************************************************************************

static int tfluids_(Main_volumetricUpSamplingNearestForward)(lua_State *L) {
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THTensor* input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));

  if (input->nDimension != 5 || output->nDimension != 5) {
    luaL_error(L, "ERROR: input and output must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zsize = input->size[2];
  const int32_t ysize = input->size[3];
  const int32_t xsize = input->size[4];

  if (output->size[0] != nbatch || output->size[1] != nfeat ||
      output->size[2] != zsize * ratio || output->size[3] != ysize * ratio ||
      output->size[4] != xsize * ratio) {
    luaL_error(L, "ERROR: input : output size mismatch.");
  }

  const real* input_data = THTensor_(data)(input);
  real* output_data = THTensor_(data)(output);

  int32_t b, f, z, y, x;
#pragma omp parallel for private(b, f, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (f = 0; f < nfeat; f++) {
      for (z = 0; z < zsize * ratio; z++) {
        for (y = 0; y < ysize * ratio; y++) {
          for (x = 0; x < xsize * ratio; x++) {
            const int64_t iout = output->stride[0] * b + output->stride[1] * f +
                output->stride[2] * z +
                output->stride[3] * y + 
                output->stride[4] * x;
            const int64_t iin = input->stride[0] * b + input->stride[1] * f +
                input->stride[2] * (z / ratio) +
                input->stride[3] * (y / ratio) +
                input->stride[4] * (x / ratio);
            output_data[iout] = input_data[iin];
          }
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// volumetricUpsamplingNearestBackward
// *****************************************************************************

static int tfluids_(Main_volumetricUpSamplingNearestBackward)(lua_State *L) {
  const int32_t ratio = static_cast<int32_t>(lua_tointeger(L, 1));
  THTensor* input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  THTensor* grad_output =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 3, torch_Tensor));
  THTensor* grad_input =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));

  if (input->nDimension != 5 || grad_output->nDimension != 5 ||
      grad_input->nDimension != 5) {
    luaL_error(L, "ERROR: input, gradOutput and gradInput must be dim 5");
  }

  const int32_t nbatch = input->size[0];
  const int32_t nfeat = input->size[1];
  const int32_t zsize = input->size[2];
  const int32_t ysize = input->size[3];
  const int32_t xsize = input->size[4];

  if (grad_output->size[0] != nbatch || grad_output->size[1] != nfeat ||
      grad_output->size[2] != zsize * ratio ||
      grad_output->size[3] != ysize * ratio ||
      grad_output->size[4] != xsize * ratio) {
    luaL_error(L, "ERROR: input : gradOutput size mismatch.");
  }

  if (grad_input->size[0] != nbatch || grad_input->size[1] != nfeat ||
      grad_input->size[2] != zsize || grad_input->size[3] != ysize ||
      grad_input->size[4] != xsize) {
    luaL_error(L, "ERROR: input : gradInput size mismatch.");
  }

  const real* input_data = THTensor_(data)(input);
  const real* grad_output_data = THTensor_(data)(grad_output);
  real * grad_input_data = THTensor_(data)(grad_input);

  int32_t b, f, z, y, x;
#pragma omp parallel for private(b, f, z, y, x) collapse(5)
  for (b = 0; b < nbatch; b++) {
    for (f = 0; f < nfeat; f++) {
      for (z = 0; z < zsize; z++) {
        for (y = 0; y < ysize; y++) {
          for (x = 0; x < xsize; x++) {
            const int64_t iout = grad_input->stride[0] * b +
                grad_input->stride[1] * f +
                grad_input->stride[2] * z +
                grad_input->stride[3] * y +
                grad_input->stride[4] * x;
            float sum = 0;
            // Now accumulate gradients from the upsampling window.
            for (int32_t zup = 0; zup < ratio; zup++) {
              for (int32_t yup = 0; yup < ratio; yup++) {
                for (int32_t xup = 0; xup < ratio; xup++) {
                  const int64_t iin = grad_output->stride[0] * b +
                      grad_output->stride[1] * f +
                      grad_output->stride[2] * (z * ratio + zup) +
                      grad_output->stride[3] * (y * ratio + yup) +
                      grad_output->stride[4] * (x * ratio + xup);
                  sum += grad_output_data[iin];
                }
              }
            }
            grad_input_data[iout] = sum;
          }
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// rectangularBlur
// *****************************************************************************

// This is a convolution with a rectangular kernel where we treat the kernel as
// an impulse train. It decouples the blur kernel size to the runtime
// resulting in an O(npix) transform (very fast even for massive kernels sizes).
static inline void DoRectangularBlurAlongAxis(
    const real* src, const int32_t size, const int32_t stride,
    const int32_t rad, real* dst) {
  // Initialize with the sum of the first pixel rad + 1 times.
  real val = src[0] * static_cast<real>(rad + 1);
  // Now accumulate the first rad - 1 elements. These two contributions
  // effectively start with the center pixel at i = -1, where we clamp the
  // edge values.
  for (int32_t i = 0; i < size && i < rad; i++) {
    val += src[i * stride];
  }

  // Now beging the algorithm.
  const real mul_const = static_cast<real>(1) / static_cast<real>(rad * 2 + 1);
  for (int32_t i = 0; i < size; i++) {
    // Move the current position over one by:
    // 1. Subtracting off the pixel 1 radius - 1 back.
    const int32_t iminus = std::max(0, i - rad - 1);
    val -= src[iminus * stride];
    // 2. Adding the pixel 1 radius forward.
    const int32_t iplus = std::min(size - 1, i + rad);
    val += src[iplus * stride];

    // Now divide by the number of output elements and set the output value.
    dst[i * stride] = val * mul_const;
  }
}

static int tfluids_(Main_rectangularBlur)(lua_State *L) {
  THTensor* src_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const int32_t blur_rad = static_cast<int32_t>(lua_tointeger(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));
  THTensor* dst_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));
  THTensor* tmp_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 5, torch_Tensor));

  if (src_tensor->nDimension != 5 || dst_tensor->nDimension != 5 ||
      tmp_tensor->nDimension != 5) {
    luaL_error(L, "ERROR: src and dst must be dim 5");
  }

  const int32_t bsize = src_tensor->size[0];
  const int32_t fsize = src_tensor->size[1];
  const int32_t zsize = src_tensor->size[2];
  const int32_t ysize = src_tensor->size[3];
  const int32_t xsize = src_tensor->size[4];

  const int32_t bstride = src_tensor->stride[0];
  const int32_t fstride = src_tensor->stride[1];
  const int32_t zstride = src_tensor->stride[2];
  const int32_t ystride = src_tensor->stride[3];
  const int32_t xstride = src_tensor->stride[4];

  const real* src = THTensor_(data)(src_tensor);
  real* dst = THTensor_(data)(dst_tensor);
  real* tmp = THTensor_(data)(tmp_tensor);

  const real* cur_src = src;
  real* cur_dst = is_3d ? dst : tmp;

  int32_t b, f, z, y, x;
  if (is_3d) {
    // Do the blur in the z-dimension.
#pragma omp parallel for private(b, f, y, x) collapse(4)
    for (b = 0; b < bsize; b++) {
      for (f = 0; f < fsize; f++) {
        for (y = 0; y < ysize; y++) {
          for (x = 0; x < xsize; x++) {
            const real* in = &cur_src[b * bstride + f * fstride + y * ystride +
                                      x * xstride];
            real* out = &cur_dst[b * bstride + f * fstride + y * ystride +
                                 x * xstride];
            DoRectangularBlurAlongAxis(in, zsize, zstride, blur_rad, out);
          }
        }
      }
    }

    cur_src = dst;
    cur_dst = tmp;
  }
  // Do the blur in the y-dimension
#pragma omp parallel for private(b, f, z, x) collapse(4)
  for (b = 0; b < bsize; b++) {
    for (f = 0; f < fsize; f++) {
      for (z = 0; z < zsize; z++) {
        for (x = 0; x < xsize; x++) {
          const real* in = &cur_src[b * bstride + f * fstride + z * zstride +
                                    x * xstride];
          real* out = &cur_dst[b * bstride + f * fstride + z * zstride +
                               x * xstride];
          DoRectangularBlurAlongAxis(in, ysize, ystride, blur_rad, out);
        }
      }
    }
  }

  cur_src = tmp;
  cur_dst = dst;

  // Do the blur in the x-dimension
#pragma omp parallel for private(b, f, z, y) collapse(4)
  for (b = 0; b < bsize; b++) {
    for (f = 0; f < fsize; f++) { 
      for (z = 0; z < zsize; z++) {
        for (y = 0; y < ysize; y++) { 
          const real* in = &cur_src[b * bstride + f * fstride + z * zstride +
                                    y * ystride];
          real* out = &cur_dst[b * bstride + f * fstride + z * zstride +
                               y * ystride];
          DoRectangularBlurAlongAxis(in, xsize, xstride, blur_rad, out);
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// signedDistanceField
// *****************************************************************************

static int tfluids_(Main_signedDistanceField)(lua_State *L) {
  THTensor* flag_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  const int32_t search_rad = static_cast<int32_t>(lua_tointeger(L, 2));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));
  THTensor* dst_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 4, torch_Tensor));

  tfluids_(FlagGrid) flags(flag_tensor, is_3d);
  tfluids_(RealGrid) dst(dst_tensor, is_3d);

  const int32_t bsize = flags.nbatch();
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();

  int32_t b, z, y, x;
#pragma omp parallel for private(b, z, y, x) collapse(4)
  for (b = 0; b < bsize; b++) {
    for (z = 0; z < zsize; z++) {
      for (y = 0; y < ysize; y++) {
        for (x = 0; x < xsize; x++) {
          if (flags.isObstacle(x, y, z, b)) {
            dst(x, y, z, b) = 0;
            continue;
          }
          real dist_sq = static_cast<real>(search_rad * search_rad);
          const int32_t zmin = std::max(0, z - search_rad);;
          const int32_t zmax = std::min(zsize - 1, z + search_rad);
          const int32_t ymin = std::max(0, y - search_rad);;
          const int32_t ymax = std::min(ysize - 1, y + search_rad);
          const int32_t xmin = std::max(0, x - search_rad);;
          const int32_t xmax = std::min(xsize - 1, x + search_rad);
          for (int32_t zsearch = zmin; zsearch <= zmax; zsearch++) {
            for (int32_t ysearch = ymin; ysearch <= ymax; ysearch++) {
              for (int32_t xsearch = xmin; xsearch <= xmax; xsearch++) {
                if (flags.isObstacle(xsearch, ysearch, zsearch, b)) {
                  const real cur_dist_sq = ((z - zsearch) * (z - zsearch) +
                                            (y - ysearch) * (y - ysearch) +
                                            (x - xsearch) * (x - xsearch));
                  if (dist_sq > cur_dist_sq) {
                    dist_sq = cur_dist_sq;
                  }
                }
              }
            }
          }
          dst(x, y, z, b) = std::sqrt(dist_sq);
        }
      }
    }
  }


  return 0;
}

// *****************************************************************************
// solveLinearSystemPCG
// *****************************************************************************

static int tfluids_(Main_solveLinearSystemPCG)(lua_State *L) {
  luaL_error(L, "ERROR: solveLinearSystemPCG not defined for CPU tensors.");
  return 0;
}

// *****************************************************************************
// solveLinearSystemJacobi
// *****************************************************************************

static int tfluids_(Main_solveLinearSystemJacobi)(lua_State *L) {
  luaL_error(L, "ERROR: solveLinearSystemJacobi not defined for CPU tensors.");
  return 0;
}

// *****************************************************************************
// normalizePressureMean
// *****************************************************************************

static int tfluids_(Main_normalizePressureMean)(lua_State *L) {
  THTensor* p_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 1, torch_Tensor));
  THTensor* flag_tensor =
      reinterpret_cast<THTensor*>(luaT_checkudata(L, 2, torch_Tensor));
  const bool is_3d = static_cast<bool>(lua_toboolean(L, 3));
  THIntTensor* inds_tensor = reinterpret_cast<THIntTensor*>(
      luaT_checkudata(L, 4, "torch.IntTensor"));

  tfluids_(FlagGrid) flags(flag_tensor, is_3d);
  tfluids_(RealGrid) pressure(p_tensor, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  // The flood-fill portion is hard to parallelize, but we'll at least do it
  // at the batch level.
  THIntTensor_resize4d(inds_tensor, nbatch, zsize, ysize, xsize);
  std::vector<std::vector<int32_t>> component_sizes;
  component_sizes.resize(nbatch);

  int32_t b;
#pragma omp parallel for private(b)
  for (b = 0; b < nbatch; b++) {
    THIntTensor* inds = THIntTensor_newSelect(inds_tensor, 0, b);
    findConnectedFluidComponents(flags, inds, b, &(component_sizes[b]));
    THIntTensor_free(inds);  // Clean up the select.
  }

  THTensor* mean_tensor = THTensor_(new)();

  int32_t z, y, x; 
  for (b = 0; b < nbatch; b++) {
    // Now calculate the component means.
    const int32_t ncomponents = component_sizes[b].size();
    THTensor_(resize1d)(mean_tensor, ncomponents);
    THTensor_(fill)(mean_tensor, (real)0);
    real* mean = THTensor_(data)(mean_tensor);
#pragma omp parallel for private(z, y, x) collapse(3)
    for (z = 0; z < zsize; z++) {
      for (y = 0; y < ysize; y++) {
        for (x = 0; x < xsize; x++) {
          const int32_t cur_component = THIntTensor_get4d(
              inds_tensor, b, z, y, x);
          // component of -1 is a non-fluid cell.
          if (cur_component >= 0) {
#pragma omp atomic
            mean[cur_component] += pressure(x, y, z, b);
          }
        }
      }
    }
    for (int32_t c = 0; c < ncomponents; c++) {
      mean[c] = mean[c] / static_cast<real>(component_sizes[b][c]);
    }

    // Now subtract the component means.
#pragma omp parallel for private(z, y, x) collapse(3)
    for (z = 0; z < zsize; z++) {
      for (y = 0; y < ysize; y++) {
        for (x = 0; x < xsize; x++) {
          const int32_t cur_component = THIntTensor_get4d(
              inds_tensor, b, z, y, x);
          if (cur_component >= 0) {
            pressure(x, y, z, b) = pressure(x, y, z, b) - mean[cur_component];
          }
        }
      }
    }
  }

  THTensor_(free)(mean_tensor);

  return 0;
}

// *****************************************************************************
// Init methods
// *****************************************************************************

static const struct luaL_Reg tfluids_(Main__) [] = {
  {"advectScalar", tfluids_(Main_advectScalar)},
  {"advectVel", tfluids_(Main_advectVel)},
  {"setWallBcsForward", tfluids_(Main_setWallBcsForward)},
  {"vorticityConfinement", tfluids_(Main_vorticityConfinement)},
  {"addBuoyancy", tfluids_(Main_addBuoyancy)},
  {"addGravity", tfluids_(Main_addGravity)},
  {"drawVelocityField", tfluids_(Main_drawVelocityField)},
  {"loadTensorTexture", tfluids_(Main_loadTensorTexture)},
  {"velocityUpdateForward", tfluids_(Main_velocityUpdateForward)},
  {"velocityUpdateBackward", tfluids_(Main_velocityUpdateBackward)},
  {"velocityDivergenceForward", tfluids_(Main_velocityDivergenceForward)},
  {"velocityDivergenceBackward", tfluids_(Main_velocityDivergenceBackward)},
  {"emptyDomain", tfluids_(Main_emptyDomain)},
  {"flagsToOccupancy", tfluids_(Main_flagsToOccupancy)},
  {"solveLinearSystemPCG", tfluids_(Main_solveLinearSystemPCG)},
  {"volumetricUpSamplingNearestForward",
   tfluids_(Main_volumetricUpSamplingNearestForward)},
  {"volumetricUpSamplingNearestBackward",
   tfluids_(Main_volumetricUpSamplingNearestBackward)},
  {"solveLinearSystemJacobi", tfluids_(Main_solveLinearSystemJacobi)},
  {"normalizePressureMean", tfluids_(Main_normalizePressureMean)},
  {"rectangularBlur", tfluids_(Main_rectangularBlur)},
  {"signedDistanceField", tfluids_(Main_signedDistanceField)},
  {NULL, NULL}  // NOLINT
};

void tfluids_(Main_init)(lua_State *L) {
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, tfluids_(Main__), "tfluids");
}

