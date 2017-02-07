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

-- This is a collection of higher level functions for moving forward the fluid
-- state to the next timestep (i.e. one integration through the PDE).
--
-- A lot of functions here are just wrappers around custom CPU or GPU modules
-- from tfluids.
--
-- The high-level function is tfluids.simulate().

local tfluids = require('tfluids')

function tfluids.getPUFlagsDensityReference(batchData)
  local p = batchData.pDiv
  local U = batchData.UDiv
  local flags = batchData.flags
  local density = batchData.density  -- Optional field.
  return p, U, flags, density
end

function tfluids.removeBCs(batch)
  batch.pBC = nil
  batch.pBCInvMask = nil
  batch.UBC = nil
  batch.UBCInvMask = nil
  batch.densityBC = nil
  batch.densityBCInvMask = nil
  collectgarbage()
end

--- @param densityVal: table of scalar values of size #density (i.e. the
--- color to set the density field). (if greyscale set to '{value}')
-- @param uScale: scalar value which sets the size of the velocity.
-- @param rad: fraction of xdim.
function tfluids.createPlumeBCs(batch, densityVal, uScale, rad)
  batch.pBC = nil  -- Nothing to do for pressure.
  batch.pBCInvMask = nil

  -- We'll set U = (0, 1, 0) in a circle on the bottom border and
  -- p = 1 in the circle.
  batch.UBC = batch.UDiv:clone():fill(0)
  batch.UBCInvMask = batch.UBC:clone():fill(1)
  assert(batch.density ~= nil,
         'plume BCs require a density field to be specified')
  assert(torch.type(densityVal) == 'table')
  if torch.isTensor(batch.density) then
    batch.densityBC = batch.density:clone():fill(0)
    batch.densityBCInvMask = batch.density:clone():fill(1)
    assert(#densityVal == 1, 'there should be a single density value')
  else
    assert(torch.type(batch.density) == 'table',
           'density should be either a table or tensor.')
    batch.densityBC = {}
    batch.densityBCInvMask = {}
    assert(#densityVal == #batch.density, 'Need a density val per channel')
    for i = 1, #batch.density do
      batch.densityBC[i] = batch.density[i]:clone():fill(0)
      batch.densityBCInvMask[i] = batch.density[i]:clone():fill(1)
    end
  end

  assert(batch.UBC:dim() == 5)
  assert(batch.UBC:size(1) == 1, 'Only single batch allowed.')
  local xdim = batch.UBC:size(5)
  local ydim = batch.UBC:size(4)
  local zdim = batch.UBC:size(3)
  local is3D = batch.UBC:size(2) == 3
  if is3D then
    assert(batch.UBC:size(2) == 3)
  else
    assert(zdim == 1)
  end
  local centerX = math.floor(xdim / 2)
  local centerZ = math.max(math.floor(zdim / 2), 1)
  local plumeRad = math.floor(xdim * rad)
  local y = 1
  local vec
  if not is3D then
    vec = torch.Tensor({0, 1}):typeAs(batch.UBC)
  else
    vec = torch.Tensor({0, 1, 0}):typeAs(batch.UBC)
  end
  vec:mul(uScale)
  for z = 1, zdim do
    for y = 1, 4 do
      for x = 1, xdim do
        local dx = centerX - x
        local dz = centerZ - z
        if (dx * dx + dz * dz) <= plumeRad * plumeRad then
          -- In the plume. Set the BCs.
          batch.UBC[{1, {}, z, y, x}]:copy(vec)
          batch.UBCInvMask[{1, {}, z, y, x}]:fill(0)
          if torch.isTensor(batch.density) then
            batch.densityBC[{1, {}, z, y, x}]:fill(densityVal[1])
            batch.densityBCInvMask[{1, {}, z, y, x}]:fill(0)
          else
            for i = 1, #batch.density do
              batch.densityBC[i][{1, {}, z, y, x}]:fill(densityVal[i])
              batch.densityBCInvMask[i][{1, {}, z, y, x}]:fill(0)
            end
          end
        else
          -- Outside the plume explicitly set the velocity to zero and leave
          -- the density alone.
          batch.UBC[{1, {}, z, y, x}]:fill(0)
          batch.UBCInvMask[{1, {}, z, y, x}]:fill(0)
        end
      end
    end
  end
end

-- We have some somewhat hacky boundary conditions, where we freeze certain
-- values on every iteration of the solver. It is equivalent to setting internal
-- fluid cells to not receive updates during the pressure projection. Note that
-- it might actually result in divergence that is never corrected (although
-- this is usually what we use it for).
local function setConstVals(batch, p, U, flags, density)
  -- Apply the external BCs.
  -- TODO(tompson): We have a separate "mask" tensor for every boundary
  -- condition type. These should really be an bit enum to specify which
  -- conditions are being set (i.e. 0x0110 would be specify U and flags only).
  -- But torch doesn't support bitwise operations natively.
  if batch.pBC ~= nil or batch.pBCInvMask ~= nil then
    -- Zero out the p values on the BCs.
    p:cmul(batch.pBCInvMask)
    -- Add back the values we want to specify.
    p:add(batch.pBC)
  end
  if batch.UBC ~= nil or batch.UBCInvMask ~= nil then
    U:cmul(batch.UBCInvMask)
    U:add(batch.UBC)
  end
  if batch.densityBC ~= nil or batch.densityBCInvMask ~= nil then
    assert(torch.type(density) == torch.type(batch.densityBC))
    if torch.isTensor(density) then
      density:cmul(batch.densityBCInvMask)
      density:add(batch.densityBC)
    else
      assert(torch.type(batch.density) == 'table')
      assert(#density == #batch.densityBC)
      for i = 1, #density do
        density[i]:cmul(batch.densityBCInvMask[i])
        density[i]:add(batch.densityBC[i])
      end
    end
  end
end

-- The top level simulation loop.
--
-- Note that this should follow the same function calls as are in
-- manta/scenes/_trainingData.py, with one exception. The setWallBcs call before
-- and after the pressure solve are performed in our model.
--
-- @param conf: config table.
-- @param mconf: model config table.
-- @param batch: CPU or GPU batch data (holds simulation state before and after
-- simulate is called).
-- @param model: The pressure model.
-- @param outputDiv: OPTIONAL. Return just before solving for pressure (i.e.
-- leave the state as UDiv and pDiv (before subtracting divergence).
function tfluids.simulate(conf, mconf, batch, model, outputDiv)
  if outputDiv == nil then
    outputDiv = false
  end

  local p, U, flags, density = tfluids.getPUFlagsDensityReference(batch)

  -- First advect all scalar fields (density, temperature, etc).
  if density ~= nil then
    if type(density) == 'table' then
      -- Density is multi-channel, i.e. for RGB densities in the 2D demo.
      for _, chan in pairs(density) do
        tfluids.advectScalar(mconf.dt, chan, U, flags, mconf.advectionMethod)
      end
    else
      tfluids.advectScalar(mconf.dt, density, U, flags, mconf.advectionMethod)
    end
  end

  -- Now self-advect velocity (must be advected last).
  tfluids.advectVel(mconf.dt, U, flags, mconf.advectionMethod)

  -- Set the manual boundary conditions.
  setConstVals(batch, p, U, flags, density)

  -- Add external forces (buoyancy and gravity).
  if density ~= nil and mconf.buoyancyScale > 0 then
    local gravity = U:new():resize(3)
    gravity[2] = (-tfluids.getDx(flags) / 4) * mconf.buoyancyScale
    if type(density) == 'table' then
      -- Just use the first channel (TODO(tompson): average the channels)
      tfluids.addBuoyancy(U, flags, density[1], gravity, mconf.dt)
    else
      tfluids.addBuoyancy(U, flags, density, gravity, mconf.dt)
    end
  end

  -- TODO(tompson): Add support for gravity.

  -- Add vorticity confinement.
  if mconf.vorticityConfinementAmp > 0 then
    local amp = tfluids.getDx(flags) * mconf.vorticityConfinementAmp
    tfluids.vorticityConfinement(U, flags, mconf.vorticityConfinementAmp)
  end

  if outputDiv then
    -- We return here during training when we WANT to train on a new
    -- divergent velocity and pressure.
    return
  end

  -- Set the constant domain values.
  setConstVals(batch, p, U, flags, density)

  if mconf.simMethod == nil or mconf.simMethod == 'convnet' then
    -- FPROP the model to perform the pressure projection & velocity
    -- calculation.
    -- NOTE: setWallBcs is performed BEFORE AND AFTER the pressure projection.
    -- So there's no need to call it again.
    local modelOutput = model:forward(torch.getModelInput(batch))
  
    -- Copy the final p and U back to the state tensor.
    local pPred, UPred = torch.parseModelOutput(modelOutput)
    p:copy(pPred)
    U:copy(UPred)
  else
    -- Calculate the RHS of the linear system (divergence).
    batch.div = batch.div or p:clone()
    batch.div:typeAs(U):resizeAs(p)
    tfluids.velocityDivergenceForward(U, flags, batch.div)

    -- Solve for pressure.
    local residual
    if mconf.simMethod == 'pcg' then
      local tol = 1e-4
      local maxIter = 100
      local precondType = 'ic0'  -- options: 'ic0', 'none', 'ilu0'
      residual = tfluids.solveLinearSystemPCG(
          p, flags, batch.div, mconf.is3D, tol, maxIter, precondType)
    elseif mconf.simMethod == 'jacobi' then
      local pTol = 0  -- Essentially, this means a fixed number of iter.
      local maxIter = 100  -- It has VERY slow convergence. Run for long time.
      residual = tfluids.solveLinearSystemJacobi(
          p, flags, batch.div, mconf.is3D, pTol, maxIter)
    else
      error('mconf.simMethod (' .. mconf.simMethod .. ') is not a valid option')
    end

    -- Now update velocity (using the pressure gradient).
    tfluids.velocityUpdateForward(U, flags, p)
  end

  -- Set the constant domain values.
  setConstVals(batch, p, U, flags, density)

  -- Finally, clamp the velocity so that even if the sim blows up it wont blow
  -- up to inf. If the velocity is this big we have other problems other than
  -- amplitude truncation).
  U:clamp(-1e6, 1e6)
end
