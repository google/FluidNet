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

local torch = require('torch')
local ok, cutorch = pcall(require, 'cutorch')
if not ok then
  print('WARNING: Couldnt load cutorch, CUDA functionality may not work.')
end
local tfluids = require('libtfluids')

tfluids._tmp = {}  -- We'll allocate temporary arrays in here.
tfluids._tmpPCG = {}

-- Get temporary cache. Note that it will only allocate if the current cache
-- is too small.
--
-- NOTE: additional calls to getTempStorage will result in reusing the temporary
-- storage allocated for previous calls. i.e. you must call this method once
-- and only once for every call that needs temp storage.
--
-- @param tensorType - ('torch.CudaTensor', etc.) str of type to allocate.
-- @param sizes - A table of tensor sizes e.g. {{2, 3}, {4}} will allocate 2
-- tensors, one of size (2 x 3) and one of size (4).
local function getTempStorage(tensorType, sizes)
  tfluids._tmp[tensorType] = (tfluids._tmp[tensorType] or
                              torch.Tensor():type(tensorType))
  assert(torch.type(sizes) == 'table')
  local numels = {}
  local totalNumel = 0
  for i = 1, #sizes do
    numels[i] = 1
    for j = 1, #(sizes[i]) do
      assert(sizes[i][j] > 0, 'tensor sizes must be positive non-zero')
      numels[i] = numels[i] * sizes[i][j]
    end
    totalNumel = totalNumel + numels[i]
  end

  tfluids._tmp[tensorType]:resize(totalNumel)

  -- Now select the sub-tensors within the large cache array.
  local ret = {}
  local offset = 1
  for i = 1, #sizes do
    local cur_tensor = tfluids._tmp[tensorType][{{offset,
                                                  offset + numels[i] - 1}}]
    cur_tensor = cur_tensor:view(unpack(sizes[i]))
    ret[i] = cur_tensor
    offset = offset + numels[i]
  end

  return ret
end

-- Advect scalar field 'p' by the input vel field 'u'.
--
-- @param dt - timestep (seconds).
-- @param s - input scalar field to advect
-- @param U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param method - OPTIONAL - "euler", "maccormack", "rk2Ours",
-- "eulerOurs", "rk3Ours", "maccormackOurs". You should use the "xxxOurs methods
-- as we better handle boundary conditions. You can use the Manta
-- implementations if you find ours are too slow.
-- @param sDst - OPTIONAL - If non-nil then this will be the returned
-- scalar field. Otherwise advection will be performed in-place.
-- @param sampleOutsideFluid - OPTIONAL - For density advection we do not want
-- to advect values inside non-fluid cells and so this should be set to false.
-- For other quantities (like temperature), this should be true.
-- Note that Manta's methods ALWAYS sample into non-fluid cells (so this only
-- applies to our routines).
-- @param maccormackStrength - OPTIONAL - (default 0.75) A strength parameter
-- will make the advection eularian (with values interpolating in between). A
-- value of 1 (which implements the update from An Unconditionally Stable
-- MaCormack Method) tends to add too much high-frequency detail (especially
-- when using Manta's maccormack implementation).
-- @param boundaryWidth - OPTIONAL - boundary width. (default 1)
local function advectScalar(dt, s, U, flags, method, sDst,
                            sampleOutsideFluid, maccormackStrength,
                            boundaryWidth)
  method = method or "maccormackOurs"
  boundaryWidth = boundaryWidth or 1
  if sampleOutsideFluid == nil then
    sampleOutsideFluid = false
  end
  maccormackStrength = maccormackStrength or 0.75

  -- Check arguments here (it's easier from lua).
  assert(s:dim() == 5 and U:dim() == 5 and flags:dim() == 5,
         'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)
  assert(s:isSameSizeAs(flags), 'Size mismatch')

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')

  -- Note: the C++ code actually does not need the input to be contiguous but
  -- the CUDA code does. But we'll conservatively constrain both anyway.
  assert(s:isContiguous() and U:isContiguous() and flags:isContiguous(),
         'Input is not contiguous')

  local tmp
  -- If we're using maccormack advection we need a temporary array, however
  -- we should just allocate always since it makes the C++ logic easier. If
  -- might also need temporary storage if the user wants to do advection
  -- in-place.
  local tmpSizes = {{bsz, 1, d, h, w}, {bsz, 1, d, h, w},
                    {bsz, U:size(2), d, h, w}, {bsz, U:size(2), d, h, w}}
  if sDst == nil then
    tmpSizes[5] = {bsz, 1, d, h, w}
  else
    assert(sDst:dim() == 5, 'Size mismatch')
    assert(sDst:isContiguous(), 'Input is not contiguous')
    assert(sDst:isSameSizeAs(s), 'Size mismatch')
  end
  tmp = getTempStorage(s:type(), tmpSizes)
  local fwd = tmp[1]   -- For method == maccormack or maccormackOurs
  local bwd = tmp[2]
  local fwdPos = tmp[3]  -- For method == maccormackOurs
  local bwdPos = tmp[4]

  s.tfluids.advectScalar(dt, s, U, flags, fwd, bwd, is3D, method,
                         fwdPos, bwdPos, boundaryWidth, sampleOutsideFluid,
                         maccormackStrength, sDst or tmp[5])
  if sDst == nil then
    -- Copy the output scalar field back to the input.
    s:copy(tmp[5])
  end
end
rawset(tfluids, 'advectScalar', advectScalar)

-- Advect velocity field 'u' by itself and store in uDst.
--
-- @param dt - timestep (seconds).
-- @param U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param method - OPTIONAL - "euler", "eulerOurs", "maccormack",
-- "maccormackOurs" (default). A value of "rk2Ours" and "rk3Ours" will use
-- maccormackOurs instead (they are left as placeholder options).
-- better handle boundary conditions. You can use the Manta implementations if
-- you find ours are too slow.
-- @param UDst - OPTIONAL - If non-nil then this will be the returned
-- velocity field. Otherwise advection will be performed in-place.
-- @param maccormackStrength - OPTIONAL - (default 0.75) A strength parameter
-- will make the advection more 1st ordre (with values interpolating in
-- between). A value of 1 (which implements the update from "An Unconditionally
-- Stable MaCormack Method") tends to add too much high-frequency detail
-- (especially when using Manta's maccormack implementation).
-- @param boundaryWidth - OPTIONAL - boundary width. (default 1)
local function advectVel(dt, U, flags, method, UDst, maccormackStrength,
                         boundaryWidth)
  method = method or "maccormackOurs"
  boundaryWidth = boundaryWidth or 1
  maccormackStrength = maccormackStrength or 0.75

  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5, 'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')
  assert(U:isContiguous() and flags:isContiguous(), 'Input is not contiguous')

  local tmp
  -- If we're using maccormack advection we need a temporary array, however
  -- we should just allocate always since it makes the C++ logic easier. If
  -- might also need temporary storage if the user wants to do advection
  -- in-place.
  if UDst == nil then
    tmp = getTempStorage(U:type(), {{bsz, U:size(2), d, h, w},
                                    {bsz, U:size(2), d, h, w},
                                    {bsz, U:size(2), d, h, w}})
  else
    tmp = getTempStorage(U:type(), {{bsz, U:size(2), d, h, w},
                                    {bsz, U:size(2), d, h, w}})
    assert(UDst:dim() == 5, 'Size mismatch')
    assert(UDst:isContiguous(), 'Input is not contiguous')
    assert(UDst:isSameSizeAs(U), 'Size mismatch')
  end
  local fwd = tmp[1]
  local bwd = tmp[2]

  U.tfluids.advectVel(dt, U, flags, fwd, bwd, is3D, method,
                      boundaryWidth, maccormackStrength, UDst or tmp[3])

  if UDst == nil then
    -- Copy the output velocity field back to the input.
    U:copy(tmp[3])
  end
end
rawset(tfluids, 'advectVel', advectVel)

-- Enforce boundary conditions on velocity MAC Grid (i.e. set slip components).
-- Note: there is no explicit "backward" method for setWallBcs since we'll apply
-- the forward function during BPROP. See set_wall_bcs.lua for more details.
--
-- @param U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
local function setWallBcsForward(U, flags)
  assert(U:dim() == 5 and flags:dim() == 5, 'Dimension mismatch')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)
  assert(flags:size(2) == 1, 'Flags must be scalar!')

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous())

  U.tfluids.setWallBcsForward(U, flags, is3D)
end
rawset(tfluids, 'setWallBcsForward', setWallBcsForward)

-- Calculate the velocity divergence (with boundary cond modifications). This is
-- essentially a replica of makeRhs in Manta.
--
-- @param U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param UDiv - output divergence (scalar field). 
local function velocityDivergenceForward(U, flags, UDiv)
  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5 and UDiv:dim() == 5,
         'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')
  assert(flags:isSameSizeAs(UDiv), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous() and UDiv:isContiguous(),
         'Input is not contiguous')

  U.tfluids.velocityDivergenceForward(U, flags, UDiv, is3D)
end
rawset(tfluids, 'velocityDivergenceForward', velocityDivergenceForward)

-- Calculates the partial derivative of the forward function.
--
-- @param U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param gradOutput - Output gradient.
-- @param gradU - return input gradient.
local function velocityDivergenceBackward(U, flags, gradOutput, gradU)
  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5 and gradOutput:dim() == 5 and
         gradU:dim() == 5, 'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')
  assert(gradU:isSameSizeAs(U), 'Size mismatch')
  assert(gradOutput:isSameSizeAs(flags), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous() and
         gradOutput:isContiguous() and gradU:isContiguous(),
         'Input is not contiguous')

  U.tfluids.velocityDivergenceBackward(U, flags, gradOutput, is3D, gradU)
end
rawset(tfluids, 'velocityDivergenceBackward', velocityDivergenceBackward)

-- Calculate the pressure gradient and subtract it into (i.e. calculate
-- U' = U - grad(p)). Some care must be taken with handling boundary conditions.
-- This function mimics correctVelocity in Manta.
-- NOTE: velocity update is done IN-PLACE.
--
-- @param U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param p - scalar pressure field.
local function velocityUpdateForward(U, flags, p)
  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5 and p:dim() == 5,
         'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')
  assert(p:isSameSizeAs(flags), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous() and p:isContiguous(),
         'Input is not contiguous')

  U.tfluids.velocityUpdateForward(U, flags, p, is3D)
 
end
rawset(tfluids, 'velocityUpdateForward', velocityUpdateForward)

-- Calculate the partial derivative of the forward function.
--
-- @param U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param p - scalar pressure field.
-- gradOutput - Mac grid output gradient.
-- gradP - scalar input gradient.
local function velocityUpdateBackward(U, flags, p, gradOutput, gradP)
  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5 and p:dim() == 5 and
         gradOutput:dim() == 5 and gradP:dim() == 5, 'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')
  assert(gradP:isSameSizeAs(p), 'Size mismatch')
  assert(gradOutput:isSameSizeAs(U), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous() and p:isContiguous() and
         gradOutput:isContiguous() and gradP:isContiguous(),
         'Input is not contiguous')

  U.tfluids.velocityUpdateBackward(U, flags, p, gradOutput, is3D, gradP)
end
rawset(tfluids, 'velocityUpdateBackward', velocityUpdateBackward)

-- Add vorticity confinement. Note: you will need to fold in dt into the
-- strength term since vort conf is actually a force. We keep this API because
-- Manta also doesn't use dt in the strength.
-- Note: Vorticity confinement is added IN-PLACE.
--
-- @param U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param strength - number indicating multiplier of force.
local function vorticityConfinement(U, flags, strength)
  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5, 'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3) 
  local h = flags:size(4)
  local w = flags:size(5)

  local is3D = U:size(2) == 3 
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous(), 'Input is not contiguous')
  assert(torch.type(strength) == 'number')

  -- Vorticity confinement needs a lot of temporary storage because we do many
  -- passes through the MAC grid. This is largely to do with the fact that
  -- Calculating force components requires sampling the input vel in many
  -- places to calculate a centered force, then applying it likewise requires
  -- many samplings of the vel field.
  local tmp = getTempStorage(U:type(), {{bsz, U:size(2), d, h, w},
                                        {bsz, 3, d, h, w},  -- always 3D
                                        {bsz, 1, d, h, w},  -- scalar
                                        {bsz, U:size(2), d, h, w}})
  local centered = tmp[1]
  local curl = tmp[2]
  local curlNorm = tmp[3]
  local force = tmp[4]
  
  U.tfluids.vorticityConfinement(U, flags, strength, centered, curl, curlNorm,
                                 force, is3D)
end
rawset(tfluids, 'vorticityConfinement', vorticityConfinement)

-- Add buoyancy force. Note: Unlike vorticityConfinement, addBuoyancy has a dt
-- term.
-- Note: Buoyancy is added IN-PLACE.
--
-- @param U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param density - scalar density grid.
-- @param gravity - 3D vector indicating direction of gravity.
-- @param dt - scalar timestep.
local function addBuoyancy(U, flags, density, gravity, dt)
  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5 and density:dim() == 5,
         'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3) 
  local h = flags:size(4) 
  local w = flags:size(5)

  local is3D = U:size(2) == 3 
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')
  assert(density:isSameSizeAs(flags), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous() and density:isContiguous(),
         'Input is not contiguous')
  assert(torch.isTensor(gravity) and gravity:dim() == 1 and
         gravity:size(1) == 3, 'gravity must be a 3D vector (even in 2D).')
  assert(torch.type(dt) == 'number', 'time step must be a number')  

  local strength = getTempStorage(U:type(), {{3}})[1]

  U.tfluids.addBuoyancy(U, flags, density, gravity, strength, dt, is3D)
end
rawset(tfluids, 'addBuoyancy', addBuoyancy)

-- Add gravity force. Note: Unlike vorticityConfinement, addGravity has a dt
-- term.
-- Note: gravity is added IN-PLACE.
--
-- @param U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
-- @param flags - input occupancy grid
-- @param gravity - 3D vector indicating direction of gravity.
-- @param dt - scalar timestep.
local function addGravity(U, flags, gravity, dt)
  -- Check arguments here (it's easier from lua).
  assert(U:dim() == 5 and flags:dim() == 5, 'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)

  local is3D = U:size(2) == 3
  if not is3D then
    assert(d == 1, '2D velocity field but zdepth > 1')
    assert(U:size(2) == 2, '2D velocity field must have only 2 channels')
  end
  assert((U:size(1) == bsz and U:size(3) == d and U:size(4) == h and
          U:size(5) == w), 'Size mismatch')

  assert(U:isContiguous() and flags:isContiguous(), 'Input is not contiguous')
  assert(torch.isTensor(gravity) and gravity:dim() == 1 and
         gravity:size(1) == 3, 'gravity must be a 3D vector (even in 2D).')
  assert(torch.type(dt) == 'number', 'time step must be a number')

  local force = getTempStorage(U:type(), {{3}})[1]  -- Used in cuda version.

  U.tfluids.addGravity(U, flags, gravity, dt, is3D, force)
end   
rawset(tfluids, 'addGravity', addGravity)

-- flipY will render the field upside down (required to get alignment with
-- grid = 0 on the bottom of the OpenGL context).
local function drawVelocityField(U, flipY)
  if flipY == nil then
    flipY = false
  end
  assert(U:dim() == 5)  -- Expected batch input.
  assert(U:size(2) == 2 or U:size(2) == 3)
  local is3D = U:size(2) == 3
  U.tfluids.drawVelocityField(U, flipY, is3D)
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

-- emptyDomain creates with TypeFluid everywhere except on the
-- border, where it is TypeObstacle. It is an easy shorthand for creating an
-- empty domain like the one used in our _trainingData.py routine.
--
-- Note: even if you don't ask for it, Manta uses a border! This is because
-- it makes their FD and adjacent cell lookup code easier. Essentially they add
-- a border of 1 that they sample from but never update during the simulation.
--
-- @param flags - A pre allocated array. Will set the domain in place in here.
-- @param is3D - If the domain is 3D or not (if it is not 3D we wont touch the
-- z border).
-- @param bnd - (OPTIONAL) the size of the boundary.
local function emptyDomain(flags, is3D, bnd)
  bnd = bnd or 1  -- You shouldn't ever make this 0.
  assert(flags:dim() == 5, 'Flags should be 5D')
  assert(flags:size(2) == 1, 'Flags should be a scalar')
  assert(((not is3D or flags:size(3) >= bnd * 2 + 1) and
          flags:size(4) >= bnd * 2 + 1 and flags:size(5) >= bnd * 2 + 1),
         'simulation domain not big enough!')
  flags.tfluids.emptyDomain(flags, is3D, bnd)
  return flags
end
rawset(tfluids, 'emptyDomain', emptyDomain)

-- getDx mimics the Simulator::getDx method in Manta.
-- Since we don't have a simulator object we infer the size of the domain from
-- the flags tensor.
local function getDx(flags)
  local gridSizeMax = math.max(math.max(flags:size(3), flags:size(4)),
                               flags:size(5))
  return 1.0 / gridSizeMax
end
rawset(tfluids, 'getDx', getDx)

-- flagsToOccupancy creates a [0, 1] occupancy grid where zero values
-- are tfluids.CellType.TypeFluid and one values are
-- tfluids.CellType.TypeObstacle. If any other cell types are found an error is
-- raised.
local function flagsToOccupancy(flags, occupancy)
  assert(flags:dim() == 5 and occupancy:dim() == 5)
  assert(flags:size(2) == 1 and flags:isSameSizeAs(occupancy))
  flags.tfluids.flagsToOccupancy(flags, occupancy)
end
rawset(tfluids, 'flagsToOccupancy', flagsToOccupancy)

-- rectangularBlur - A fast impulse based rectangular blur method. It is O(n)
-- where n is the number of pixels and has runtime independent of the kernel
-- size.
--
-- Works on 3D or 2D grids.
local function rectangularBlur(src, blurRad, is3D, dst)
  assert(src:dim() == 5 and dst:dim() == 5)
  assert(src:isSameSizeAs(dst))
  assert(blurRad > 0 and math.floor(blurRad) == blurRad,
         'blurRad must be a positive, non-zero integer')
  assert(src:isContiguous() and dst:isContiguous())

  -- We need a temp buffer so we don't destroy src (we do separable
  -- convolutions so we need an intermediate buffer).
  local tmp = getTempStorage(src:type(), {src:size():totable()})[1]

  dst.tfluids.rectangularBlur(src, blurRad, is3D, dst, tmp)
end
rawset(tfluids, 'rectangularBlur', rectangularBlur)

-- signedDistanceField calculates the signed distance field of the input image.
-- NOTE: This is NOT a linear time signed distance transform. It is O(pix * 
-- searchRad^n). We simply look in a local window for obstacle cells and collect
-- the min distance. All distances above searchRad are clamped (to searchRad).
--
-- Works on 3D or 2D grids.
local function signedDistanceField(flags, searchRad, is3D, dst)
  assert(flags:dim() == 5 and dst:dim() == 5)
  assert(flags:isSameSizeAs(dst))
  assert(flags:isContiguous() and dst:isContiguous())
  assert(flags:size(2) == 1, 'flags must be scalar')
  assert(searchRad > 0 and math.floor(searchRad) == searchRad,
         'searchRad must be a positive, non-zero integer')

  dst.tfluids.signedDistanceField(flags, searchRad, is3D, dst)
end
rawset(tfluids, 'signedDistanceField', signedDistanceField)

-- volumetricUpSamplingNearestForward - This lua interface is just for testing.
-- you should use the Module.
local function volumetricUpSamplingNearestForward(ratio, input, output)
  input.tfluids.volumetricUpSamplingNearestForward(ratio, input, output)
end
rawset(tfluids, 'volumetricUpSamplingNearestForward',
       volumetricUpSamplingNearestForward)
local function volumetricUpSamplingNearestBackward(ratio, input, gOut, gIn)
  input.tfluids.volumetricUpSamplingNearestBackward(ratio, input, gOut, gIn)
end
rawset(tfluids, 'volumetricUpSamplingNearestBackward',
       volumetricUpSamplingNearestBackward)

-- Solve the linear system using cusparse's PCG method.
-- Note: Since we don't receive a velocity field, we need to receive the is3D
-- flag from the caller.
--
-- @param p: The output pressure field (i.e. the solution to A * p = div)
-- @param flags: The input flag grid.
-- @param div: The velocity divergence.
-- @param is3D: If true then we expect a 3D domain.
-- @param tol: OPTIONAL (default = 1e-6), PCG termination tollerance.
-- @param maxIter: OPTIONAL (default = 1000), max number of PCG iterations.
-- @param precondType:  OPTIONAL (default = 'ic0'). Use preconditioning.
-- Options are: 'none', 'ilu0' (incomplete LU rank 0), 'ic0' (incomplete
-- Cholesky rank 0).
-- @param verbose: OPTIONAL (default = false), if true print out iteration res.
--
-- @return: the max residual of the PCG solver across all batches.
local function solveLinearSystemPCG(p, flags, div, is3D, tol, maxIter,
                                    precondType, verbose)
  -- Check arguments here (it's easier from lua).
  assert(p:dim() == 5 and flags:dim() == 5 and div:dim() == 5,
         'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)
  assert(p:isSameSizeAs(flags), 'size mismatch')
  assert(div:isSameSizeAs(flags), 'size mismatch')
  if not is3D then
    assert(d == 1, 'd > 1 for a 2D domain')
  end

  if verbose == nil then
    verbose = false
  end
  precondType = precondType or 'ic0';
  tol = tol or 1e-6
  maxIter = maxIter or 1000

  assert(torch.type(p) == 'torch.CudaTensor', 'Only CUDA is supported for now')
  assert(p:isContiguous() and flags:isContiguous() and div:isContiguous())

  -- Note: we pass in the tfluids._tmpPCG table so that we can reuse static
  -- tensors allocated within the table.
  return p.tfluids.solveLinearSystemPCG(
      tfluids._tmpPCG, p, flags, div, is3D, precondType, tol, maxIter,
      verbose)
end
rawset(tfluids, 'solveLinearSystemPCG', solveLinearSystemPCG)


-- Solve the linear system using the Jacobi method.
-- Note: Since we don't receive a velocity field, we need to receive the is3D
-- flag from the caller.
--
-- @param p: The output pressure field (i.e. the solution to A * p = div)
-- @param flags: The input flag grid.
-- @param div: The velocity divergence.
-- @param is3D: If true then we expect a 3D domain.
-- @param pTol: OPTIONAL (default = 1e-5), ||p - p_prev|| termination cond.
-- @param maxIter: OPTIONAL (default = 1000), max number of PCG iterations.
-- @param verbose: OPTIONAL (default = false), if true print out iteration res.
--
-- @return: the max pTol across the batches.
local function solveLinearSystemJacobi(p, flags, div, is3D, pTol, maxIter,
                                       verbose)
  -- Check arguments here (it's easier from lua).
  assert(p:dim() == 5 and flags:dim() == 5 and div:dim() == 5,
         'Dimension mismatch')
  assert(flags:size(2) == 1, 'flags is not scalar')
  local bsz = flags:size(1)
  local d = flags:size(3)
  local h = flags:size(4)
  local w = flags:size(5)
  assert(p:isSameSizeAs(flags), 'size mismatch')
  assert(div:isSameSizeAs(flags), 'size mismatch')
  if not is3D then
    assert(d == 1, 'd > 1 for a 2D domain')
  end

  if verbose == nil then
    verbose = false
  end
  pTol = pTol or 1e-5
  maxIter = maxIter or 1000

  assert(torch.type(p) == 'torch.CudaTensor', 'Only CUDA is supported for now')
  assert(p:isContiguous() and flags:isContiguous() and div:isContiguous())

  -- We need some temporary storage to avoid dynamic allocations.
  local tmp = getTempStorage(p:type(), {{bsz, 1, d, h, w},
                                        {bsz, 1, d, h, w},
                                        {bsz}})
  local pPrev = tmp[1]
  local pDelta = tmp[2]
  local pDeltaNorm = tmp[3]

  local residual = p.tfluids.solveLinearSystemJacobi(
      p, flags, div, pPrev, pDelta, pDeltaNorm, is3D, pTol, maxIter, verbose)

  -- Note: unlike PCG, we do not ensure that each connected component of fluid
  -- cells has zero mean. If this is necessary you should call
  -- tfluids.normalizePressureMean explicitly (it's somewhat expensive).

  return residual
end
rawset(tfluids, 'solveLinearSystemJacobi', solveLinearSystemJacobi)

-- This function will transfer to the CPU if p and flags are cuda tensors (and
-- sync back). Use sparingly. TODO(tompson): do flood fill on the GPU.
--
-- We will perform a connected-component search (using flood-fill) to find the
-- connected components of fluid cells, and we will then subtract the mean of
-- each component. This removes the arbitrary DC bias in each connected
-- component.
--
-- @param p: The output pressure field (i.e. the solution to A * p = div)
-- @param flags: The input flag grid.
local function normalizePressureMean(p, flags, is3D)
  local pCPU = p
  local flagsCPU = flags
  if torch.type(p) == 'torch.CudaTensor' then  -- Assume cuda, float or double.
    -- GPU to CPU sync.
    pCPU = p:float()
    flagsCPU = flags:float()
  end

  -- Need temp space for the index list.
  local inds = getTempStorage('torch.IntTensor', {pCPU:size():totable()})[1]

  pCPU.tfluids.normalizePressureMean(pCPU, flagsCPU, is3D, inds)

  if torch.type(p) == 'torch.CudaTensor' then
    -- CPU to GPU sync.
    p:copy(pCPU)
  end
end
rawset(tfluids, 'normalizePressureMean', normalizePressureMean)

-- Now include the modules.
include('flags_to_occupancy.lua')
include('set_wall_bcs.lua')
include('velocity_divergence.lua')
include('velocity_update.lua')
include('volumetric_up_sampling_nearest.lua')

-- Also include the test framework.
include('test_tfluids.lua')

return tfluids
