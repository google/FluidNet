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

function tfluids.getPUGeomDensityReference(batchData)
  local p = batchData.pDiv[{{}, 1}]  -- Remove the unary dimension.
  local U = batchData.UDiv
  local geom = batchData.geom[{{}, 1}]  -- Remove the unary dimension.
  local density = batchData.density  -- Optional field.
  return p, U, geom, density
end

function tfluids.removeBCs(batch)
  batch.pBC = nil
  batch.pBCInvMask = nil
  batch.geomBC = nil
  batch.geomBCInvMask = nil
  batch.UBC = nil
  batch.UBCInvMask = nil
  batch.densityBC = nil
  batch.densityBCInvMask = nil
end

-- @param densityVal: table of scalar values of size density:size(2) (i.e. the
-- color to set the density field).
-- @param uScale: scalar value which sets the size of the velocity.
-- @param rad: fraction of xdim.
function tfluids.createPlumeBCs(batch, densityVal, uScale, rad)
  batch.pBC = nil  -- Nothing to do for pressure.
  batch.pBCInvMask = nil
  batch.geomBC = nil  -- Nothing to do for geometry.
  batch.geomBCInvMask = nil

  -- We'll set U = (0, 1, 0) in a circle on the bottom border and
  -- p = 1 in the circle.
  batch.UBC = batch.UDiv:clone():fill(0)
  batch.UBCInvMask = batch.UBC:clone():fill(1)
  assert(batch.density ~= nil,
         'plume BCs require a density field to be specified')
  batch.densityBC = batch.density:clone():fill(0)
  batch.densityBCInvMask = batch.density:clone():fill(1)

  assert(batch.UBC:dim() == 5)
  assert(batch.UBC:size(1) == 1, 'Only single batch allowed.')
  local xdim = batch.UBC:size(5)
  local ydim = batch.UBC:size(4)
  local zdim = batch.UBC:size(3)
  local twoDim = batch.UBC:size(2) == 2
  if not twoDim then
    assert(batch.UBC:size(2) == 3)
  else
    assert(zdim == 1)
  end
  local centerX = math.floor(xdim / 2)
  local centerZ = math.max(math.floor(zdim / 2), 1)
  local plumeRad = math.floor(xdim * rad)
  local y = 1
  local vec
  if twoDim then
    vec = torch.Tensor({0, 1}):typeAs(batch.UBC)
  else
    vec = torch.Tensor({0, 1, 0}):typeAs(batch.UBC)
  end
  vec:mul(uScale)
  densityVal = torch.Tensor(densityVal):typeAs(batch.densityBC)
  assert(densityVal:size(1) == batch.densityBC:size(2),
         'Incorrect density specifier length')
  for z = 1, zdim do
    for y = 1, 4 do
      for x = 1, xdim do
        local dx = centerX - x
        local dz = centerZ - z
        if (dx * dx + dz * dz) <= plumeRad * plumeRad then
          -- In the plume. Set the BCs.
          batch.UBC[{1, {}, z, y, x}]:copy(vec)
          batch.UBCInvMask[{1, {}, z, y, x}]:fill(0)
          batch.densityBC[{1, {}, z, y, x}]:copy(densityVal)
          batch.densityBCInvMask[{1, {}, z, y, x}]:fill(0)
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

local function setBoundaryConditionsAverage(p, U, geom, density)
  -- This is NOT physically plausible, but helps with divergence blowup on
  -- the boundaries. The boundaries will all receive the local average.
  tfluids._UAve = tfluids._UAve or torch.Tensor():typeAs(U)
  tfluids._UAve:resizeAs(U):copy(U)
  tfluids.averageBorderCells(U, geom, tfluids._UAve)
  -- Copy the results back.
  U:copy(tfluids._UAve)
end

local function setBoundaryConditionsZero(p, U, geom, density)
  assert(U:dim() == 4)
  local twoDim = U:size(1) == 2
  -- TODO(kris): Only zero velocities exiting boundaries.
  if not twoDim then
    U[{{}, {1}, {}, {}}]:zero()
    U[{{}, {-1}, {}, {}}]:zero()
  end
  U[{{}, {}, {1}, {}}]:zero()
  U[{{}, {}, {-1}, {}}]:zero()
  U[{{}, {}, {}, {1}}]:zero()
  U[{{}, {}, {}, {-1}}]:zero()
end

local function setBoundaryConditionsBatch(batch, bndType)
  if bndType == 'None' then
    return
  end
  local p, U, geom, density = tfluids.getPUGeomDensityReference(batch)

  for b = 1, U:size(1) do
    local curP = p[b]
    local curU = U[b]
    local curGeom = geom[b]
    local curDensity
    if density ~= nil then
      curDensity = density[b]
    end
    bndType = bndType or 'Zero'
    if bndType == 'Zero' then
      setBoundaryConditionsZero(curP, curU, curGeom, curDensity)
    elseif bndType == 'Ave' then
      setBoundaryConditionsAverage(curP, curU, curGeom, curDensity)
    else
      error('Bad bndType value: ' .. bndType)
    end
    tfluids.setObstacleBcs(curU, curGeom)
  end

  -- Apply the external BCs.
  -- TODO(tompson): We have a separate "mask" tensor for every boundary
  -- condition type. These should really be an bit enum to specify which
  -- conditions are being set (i.e. 0x0110 would be specify U and geom only).
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
  if batch.geomBC ~= nil or batch.geomBCInvMask ~= nil then
    -- Not sure why we would ever need this, but leave it in there for future
    -- (unforeseen) use cases.
    geom:cmul(batch.geomBCInvMask)
    geom:add(batch.geomBC)
  end
  if batch.densityBC ~= nil or batch.densityBCInvMask ~= nil then
    density:cmul(batch.densityBCInvMask)
    density:add(batch.densityBC)
  end
end

local function advectScalarBatch(dt, scalar, U, geom, advectionMethod)
  -- Allocate a temporary output (because we cannot advect in place).
  tfluids._scalarAdv = tfluids._scalarAdv or torch.Tensor():typeAs(U)
  tfluids._scalarAdv:resizeAs(scalar[{1, 1}])

  for b = 1, scalar:size(1) do
    for c = 1, scalar:size(2) do
      -- Independently advect each channel.
      tfluids.advectScalar(dt, scalar[{b, c}], U[b], geom[b],
                           tfluids._scalarAdv, advectionMethod)
      scalar[{b, c}]:copy(tfluids._scalarAdv)
    end
  end

end

local function advectVelocityBatch(dt, U, geom, advectionMethod)
  -- Allocate a temporary output (because we cannot advect in place).
  tfluids._UAdv = tfluids._UAdv or torch.Tensor():typeAs(U)
  tfluids._UAdv:resizeAs(U[1])

  for b = 1, U:size(1) do
    tfluids.advectVel(dt, U[b], geom[b], tfluids._UAdv, advectionMethod)
    U[b]:copy(tfluids._UAdv)
  end
end

local function vorticityConfinementBatch(dt, scale, U, geom)
  assert(U:dim() == 5 and geom:dim() == 4)
  -- vorticityConfinement needs scratch space to store the curl and ||curl||.
  if tfluids._curl == nil or torch.type(tfluids._curl) ~= torch.type(U) then
    tfluids._curl = U:clone()
  end
  if U:size(2) == 2 then
    tfluids._curl:resizeAs(geom[1])  -- 2D curl is a scalar field.
  else
    tfluids._curl:resizeAs(U[1])
  end
  if tfluids._magCurl == nil or
      torch.type(tfluids._magCurl) ~= torch.type(geom) then
    tfluids._magCurl = geom:clone()
  end
  tfluids._magCurl:resizeAs(geom[1])

  for b = 1, U:size(1) do
    tfluids.vorticityConfinement(dt, scale, U[b], geom[b], tfluids._curl,
                                 tfluids._magCurl)
  end
end

-- NOTE: this function is buoyancy for GASSES. It does not model buoyancy of
-- fluid / air interactions.
function tfluids.buoyancyBatch(dt, scalar, U, geom, scale)
  -- NEW BUOYANCY CODE THAT MATCHES MANTA, note that it's not really correct.
  -- i.e. it doesn't really follow the buoyancy calc in Bridson, but I
  -- understand how they get there.
  -- This also matches the OLD buoyancy code (algebraically), but is
  -- significantly simpler.
  assert(U:dim() == 5 and geom:dim() == 4)
  assert(scalar:dim() == 5)

  tfluids._fbuoy = tfluids._fbuoy or torch.Tensor():typeAs(scalar)
  tfluids._T = tfluids._T or torch.FloatTensor():typeAs(scalar)
  tfluids._invGeom = tfluids._invGeom or torch.FloatTensor():typeAs(scalar)
  
  -- coeff: Takes into account gravity. It is derived from the effective scale
  -- of the old code, normalized by dt = 0.4.
  local coeff = 0.5470
  
  for b = 1, U:size(1) do
    tfluids._T:resizeAs(scalar[b][1])
    if scalar:size(2) == 1 then
      tfluids._T:copy(scalar[b][1])  -- temperature IS the scalar value.
    else
      torch.mean(tfluids._T, scalar[b], 1)  -- temperature is the mean.
    end
    
    tfluids._invGeom:resizeAs(geom[b])
    tfluids._invGeom:copy(geom[b]):mul(-1):add(1)
    tfluids._T:cmul(tfluids._invGeom)

    -- From Bridson page 102: 
    --   --> Fbuoy / -g = coeffA * s + -coeffB * (T - Tamb)
    -- If we set Tamb to zero and assume s = T then we just get a simple scalar
    -- of the temperature pointing up (because it's negative of gravity).
    tfluids._fbuoy:resizeAs(tfluids._T)
    tfluids._fbuoy:zero()
    tfluids._fbuoy:add(coeff * dt * scale, tfluids._T)
    tfluids._fbuoy:cmul(tfluids._invGeom)  -- Redundant, but keep for clarity.

    -- Finally apply to the y-component of velocity. 
    U[{b, 2}]:add(tfluids._fbuoy)
  end
end

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

  local p, U, geom, density = tfluids.getPUGeomDensityReference(batch)

  -- First advect all scalar fields (density, temperature, etc).
  if density ~= nil then
    advectScalarBatch(mconf.dt, density, U, geom, mconf.advectionMethod)
  end

  -- Now self-advect velocity (must be advected last).
  advectVelocityBatch(mconf.dt, U, geom, mconf.advectionMethod)
  setBoundaryConditionsBatch(batch, mconf.bndType)

  -- Add external forces (buoyancy and gravity).
  if density ~= nil and mconf.buoyancyScale > 0 then
    tfluids.buoyancyBatch(mconf.dt, density, U, geom, mconf.buoyancyScale)
    setBoundaryConditionsBatch(batch, mconf.bndType)
  end

  -- TODO(tompson,kris): Add support for gravity (easy to do, just add (-dt) * g
  -- to the velocity field y component).

  -- Add vorticity confinement.
  if mconf.vorticityConfinementAmp > 0 then
    vorticityConfinementBatch(mconf.dt, mconf.vorticityConfinementAmp, U, geom)
    setBoundaryConditionsBatch(batch, mconf.bndType)
  end

  if outputDiv then
    -- We return here during training when we WANT to train on a new
    -- divergent velocity and pressure.
    return
  end

  -- FPROP the model to perform the pressure projection & velocity calculation.
  local modelOutput = model:forward(torch.getModelInput(batch))

  -- Copy the final p and U back to the state tensor.
  local pPred, UPred = torch.parseModelOutput(modelOutput)
  p:copy(pPred)
  U:copy(UPred)

  setBoundaryConditionsBatch(batch, mconf.bndType)

  -- Finally, clamp the velocity so that even if the sim blows up it wont blow
  -- up to inf (which causes some of our kernels to hang infinitely).
  -- (if the velocity is this big we have other problems other than truncation).
  U:clamp(-1e6, 1e6)
end
