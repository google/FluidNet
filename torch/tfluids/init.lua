-- Copyright 2016 Google Inc, NYU.
-- 
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
-- 
--     http://www.apache.org/licenses/LICENSE-2.0
-- 
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

local cutorch = require('cutorch')
local torch = require('torch')
local tfluids = require('libtfluids')

-- Advect scalar field 'p' by the input vel field 'u'.
--
-- @param dt - timestep (seconds).
-- @param p - input scalar field to advect, size: (depth x height x width)
-- @param u - input vel field, size: (2/3 x depth x height x width)
-- @param geom - input occupancy grid, size: (depth x height x width)
-- @param pDst - Return (pre-allocated) scalar field, same size as p.
-- @param method - OPTIONAL - "euler", "rk2" (default) or "maccormack".
-- @param sampleIntoGeom - OPTIONAL if false (default) then the scalar field
-- will be clamped at the non-geometry samples (i.e. the bilinear interpolation
-- will not sample INTO the geometry region).
local function advectScalar(dt, p, u, geom, pDst, method, sampleIntoGeom)
  if sampleIntoGeom == nil then
    sampleIntoGeom = false
  end
  method = method or "rk2"
  -- Check arguments here (it's easier from lua).
  local twoDim = u:size(1) == 2
  assert(p:dim() == 3 and u:dim() == 4 and geom:dim() == 3 and
         pDst:dim() == 3, 'Dimension mismatch')
  local d = geom:size(1)
  local h = geom:size(2)
  local w = geom:size(3)
  if twoDim then
    assert(d == 1, '2D velocity field but zdepth > 1')
  end
  assert(p:isSameSizeAs(geom) and p:isSameSizeAs(pDst), 'Size mismatch')
  assert(u:size(2) == d and u:size(3) == h and u:size(4) == w, 'Size mismatch')
  assert((twoDim and u:size(1) == 2) or (not twoDim and u:size(1) == 3))
  assert(p:isContiguous() and u:isContiguous() and geom:isContiguous() and
         pDst:isContiguous(), 'Input is not contiguous')

  p.tfluids.advectScalar(dt, p, u, geom, pDst, method, sampleIntoGeom)
end
rawset(tfluids, 'advectScalar', advectScalar)

-- Advect vel field 'u' by the input vel field 'u' (and store result in uDst).
--
-- @param dt - timestep (seconds).
-- @param u - input vel field to advect, size: (2/3 x depth x height x width).
-- @param geom - input occupancy grid, size: (depth x height x width).
-- @param uDst - Return (pre-allocated) velocity field, same size as u.
-- @param method - OPTIONAL - "euler", "rk2" (default) or "maccormack".
--
-- NOTE: advectVel does not have a "sampleIntoGeom" parameter. We also
-- sample into geometry cells when advecting the velocity field. The particle
-- trace is still clamped at the geometry border, but the final bilinear sample
-- will sample INTO the geometry voxels. This assumes you have called
-- setObstacleBCS which will fill internal geometry cell velocities so that the
-- interpolated face velocity is zero along the face normal.
local function advectVel(dt, u, geom, uDst, method)
  method = method or "rk2"
  -- Check arguments here (it's easier from lua).
  local twoDim = u:size(1) == 2
  assert(u:dim() == 4 and geom:dim() == 3 and uDst:dim() == 4,
         'Dimension mismatch')
  local d = geom:size(1)
  local h = geom:size(2)
  local w = geom:size(3)
  if twoDim then
    assert(d == 1, '2D velocity field but zdepth > 1')
  end
  assert(u:isSameSizeAs(uDst), 'Size mismatch')
  assert(u:size(2) == d and u:size(3) == h and u:size(4) == w, 'Size mismatch')
  assert((twoDim and u:size(1) == 2) or (not twoDim and u:size(1) == 3))
  assert(u:isContiguous() and geom:isContiguous() and uDst:isContiguous(),
         'Input is not contiguous')

  u.tfluids.advectVel(dt, u, geom, uDst, method)
end
rawset(tfluids, 'advectVel', advectVel)

-- Magnify the vortices in  vel field 'u' by the input vel field 'u' (and 
-- store result in uDst).
--
-- @param dt - timestep (seconds).
-- @param scale - scale of vortex magnifying force
-- @param u - input vel field to effect, size: (2/3 x depth x height x width).
-- @param geom - input occupancy grid, size: (depth x height x width)
-- @param curl - temporary buffer (stores curl of u).  size of u in 3D or size
-- of geom in 2D (curl is a scalar field for the 2D case).
-- @param magCurl - temporary buffer (stores ||curl of u||). size of geom.
local function vorticityConfinement(dt, scale, u, geom, curl, magCurl)
  -- Check arguments here (it's easier from lua).
  local twoDim = u:size(1) == 2
  assert(u:dim() == 4 and geom:dim() == 3, 'Dimension mismatch')
  local d = geom:size(1)
  local h = geom:size(2)
  local w = geom:size(3)
  assert(u:size(2) == d and u:size(3) == h and u:size(4) == w, 'Size mismatch')
  assert((twoDim and u:size(1) == 2) or (not twoDim and u:size(1) == 3))
  assert(u:isContiguous() and geom:isContiguous(), 'input is not contiguous')
  if not twoDim then
    assert(curl:isSameSizeAs(u))
  else
    assert(curl:isSameSizeAs(geom))
  end
  assert(magCurl:isSameSizeAs(geom))

  u.tfluids.vorticityConfinement(dt, scale, u, geom, curl, magCurl)
end
rawset(tfluids, 'vorticityConfinement', vorticityConfinement)

-- Do a local averaging for all the border cells.
-- @param tensor - Tensor of size (nchan x depth x height x width).
-- @param geom - occupancy grid of size (depth x height x width).
-- @param ret - Output tensor of size(tensor).
local function averageBorderCells(tensor, geom, ret)
  assert(torch.isTensor(tensor) and torch.isTensor(ret) and
         torch.isTensor(geom))
  assert(tensor:isSameSizeAs(ret), 'Size mismatch.')
  assert(tensor:dim() == 4, '4D tensor expected')
  assert(geom:dim() == 3)
  assert(geom:size(1) == ret:size(2) and geom:size(2) == ret:size(3) and
         geom:size(3) == ret:size(4))
  assert(geom:isContiguous() and tensor:isContiguous() and
         ret:isContiguous())
  tensor.tfluids.averageBorderCells(tensor, geom, ret)
end
rawset(tfluids, 'averageBorderCells', averageBorderCells)

-- Set internal obstacle boundary conditions.
-- @param U - Tensor of size (2/3 x depth x height x width).
-- @param geom - occupancy grid of size (depth x height x width).
local function setObstacleBcs(U, geom)
  assert(torch.isTensor(U) and torch.isTensor(geom))
  assert(U:dim() == 4, '4D tensor expected')
  assert(geom:dim() == 3)
  assert(geom:size(1) == U:size(2) and geom:size(2) == U:size(3) and
         geom:size(3) == U:size(4))
  assert(geom:isContiguous() and U:isContiguous())
  U.tfluids.setObstacleBcs(U, geom)
end
rawset(tfluids, 'setObstacleBcs', setObstacleBcs)


-- Interpolate field (exposed for debugging) using trilinear interpolation.
-- Can only interpolate positions within non-geometry cells (will throw an
-- error otherwise). Note: (0, 0, 0) is defined as the CENTER of the first
-- grid cell, so (-0.5, -0.5, -0.5) is the start of the 3D grid and (dimx + 0.5,
-- dimy + 0.5, dimz + 0.5) is the last cell.
-- @param field - Tensor of size (depth x height x width).
-- @param geom - occupancy grid of size (depth x height x width).
-- @param pos - Tensor of size (3).
-- @param sampleIntoGeom - OPTIONAL see argument description for advectScalar.
local function interpField(field, geom, pos, sampleIntoGeom)
  if sampleIntoGeom == nil then
    sampleIntoGeom = true
  end
  assert(torch.isTensor(field) and torch.isTensor(geom) and torch.isTensor(pos))
  assert(field:dim() == 3 and geom:dim() == 3, '4D tensor expected')
  assert(field:isSameSizeAs(geom), 'Size mismatch')
  assert(pos:dim() == 1 and pos:size(1) == 3)
  assert(pos[1] >= -0.5 and pos[1] <= geom:size(3) - 0.5,
      'pos[1] out of bounds')
  assert(pos[2] >= -0.5 and pos[2] <= geom:size(2) - 0.5,
      'pos[2] out of bounds')
  assert(pos[3] >= -0.5 and pos[3] <= geom:size(1) - 0.5,
      'pos[3] out of bounds')

  return field.tfluids.interpField(field, geom, pos, sampleIntoGeom)
end
rawset(tfluids, 'interpField', interpField)

-- flipY will render the field upside down (required to get alignment with
-- grid = 0 on the bottom of the OpenGL context).
local function drawVelocityField(U, flipY)
  if flipY == nil then
    flipY = false
  end
  assert(U:dim() == 5)  -- Expected batch input.
  assert(U:size(2) == 2 or U:size(2) == 3)
  local twoDim = U:size(2) == 2
  assert(not twoDim or U:size(3) == 1)
  U.tfluids.drawVelocityField(U, flipY)
end
rawset(tfluids, 'drawVelocityField', drawVelocityField)

-- loadTensorTexture performs a glTexImage2D call on the imTensor data (and
-- handles correct re-swizzling of the data).
local function loadTensorTexture(imTensor, texGLID, filter, flipY)
  if flipY == nil then
    flipY = true
  end
  assert(imTensor:dim() == 2 or imTensor:dim() == 3)
  imTensor.tfluids.loadTensorTexture(imTensor, texGLID, filter, flipY)
end
rawset(tfluids, 'loadTensorTexture', loadTensorTexture)

-- calcVelocityUpdate assumes the input is batched.
local function calcVelocityUpdate(deltaU, p, geom, matchManta)
  assert(deltaU:dim() == 5 and p:dim() == 4 and geom:dim() == 4)
  local nbatch = deltaU:size(1)
  local twoDim = deltaU:size(2) == 2
  local zdim = deltaU:size(3)
  local ydim = deltaU:size(4)
  local xdim = deltaU:size(5)
  if not twoDim then
    assert(deltaU:size(2) == 3, 'Bad number of velocity slices')
  end
  assert(p:isSameSizeAs(geom))
  assert(p:size(1) == nbatch)
  assert(p:size(2) == zdim)
  assert(p:size(3) == ydim)
  assert(p:size(4) == xdim)
  assert(p:isContiguous() and deltaU:isContiguous() and geom:isContiguous())
  deltaU.tfluids.calcVelocityUpdate(deltaU, p, geom, matchManta)
end
rawset(tfluids, 'calcVelocityUpdate', calcVelocityUpdate)

-- Calculates the partial derivative of calcVelocityUpdate.
local function calcVelocityUpdateBackward(gradP, p, geom, gradOutput,
                                          matchManta)
  assert(gradP:dim() == 4, 'gradP must be 4D')
  assert(p:dim() == 4, 'p must be 4D')
  assert(geom:dim() == 4, 'geom must be 4D')
  assert(gradOutput:dim() == 5, 'gradOutput must be 5D')
  assert(gradP:isSameSizeAs(p) and gradP:isSameSizeAs(geom))
  local nbatch = gradP:size(1)
  local zdim = gradP:size(2)
  local ydim = gradP:size(3)
  local xdim = gradP:size(4)
  local twoDim = gradOutput:size(2) == 2
  if not twoDim then
    assert(gradOutput:size(2) == 3, 'Bad number of velocity slices')
  else
    assert(zdim == 1, 'zdim is too large')
  end
  assert(p:isContiguous() and gradP:isContiguous() and geom:isContiguous()
         and gradOutput:isContiguous())
  gradP.tfluids.calcVelocityUpdateBackward(gradP, p, geom, gradOutput,
                                           matchManta)
end
rawset(tfluids, 'calcVelocityUpdateBackward', calcVelocityUpdateBackward)

-- calcVelocityDivergence assumes the input is batched.
local function calcVelocityDivergence(U, geom, UDiv)
  assert(U:dim() == 5 and geom:dim() == 4 and UDiv:dim() == 4)
  assert(geom:isSameSizeAs(UDiv))
  local nbatch = U:size(1)
  local twoDim = U:size(2) == 2
  local zdim = U:size(3)
  local ydim = U:size(4)
  local xdim = U:size(5)
  if not twoDim then
    assert(U:size(2) == 3, 'Bad number of velocity slices')
  end
  assert(geom:size(1) == nbatch)
  assert(geom:size(2) == zdim)
  assert(geom:size(3) == ydim)
  assert(geom:size(4) == xdim)
  assert(U:isContiguous() and geom:isContiguous())
  U.tfluids.calcVelocityDivergence(U, geom, UDiv)
end
rawset(tfluids, 'calcVelocityDivergence', calcVelocityDivergence)

-- Calculates the partial derivative of calcVelocityDivergence.
local function calcVelocityDivergenceBackward(gradU, U, geom, gradOutput)
  assert(gradU:dim() == 5 and U:dim() == 5 and geom:dim() == 4 and
         gradOutput:dim() == 4)
  assert(gradU:isSameSizeAs(U))
  assert(gradOutput:isSameSizeAs(geom))
  local nbatch = geom:size(1)
  local zdim = geom:size(2)
  local ydim = geom:size(3)
  local xdim = geom:size(4)
  local twoDim = U:size(2) == 2
  if not twoDim then
    assert(U:size(2) == 3, 'Bad number of velocity slices')
  else
    assert(zdim == 1, 'zdim is too large')
  end
  assert(U:size(1) == nbatch)
  assert(U:size(3) == zdim)
  assert(U:size(4) == ydim)
  assert(U:size(5) == xdim)
  assert(U:isContiguous() and gradU:isContiguous() and geom:isContiguous())
  U.tfluids.calcVelocityDivergenceBackward(gradU, U, geom, gradOutput)
end
rawset(tfluids, 'calcVelocityDivergenceBackward',
       calcVelocityDivergenceBackward)

-- Also include the test framework.
include('test_tfluids.lua')

return tfluids
