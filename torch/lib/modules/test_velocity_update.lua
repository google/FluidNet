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

local nn = require('nn')
local tfluids = require('tfluids')
local image = require('image')
if nn.VelocityUpdate == nil then
  dofile("velocity_update.lua")
end
if nn.InjectTensor == nil then
  dofile("inject_tensor.lua")
end
if torch.calcVelocityUpdate == nil then
  dofile("../../utils/calc_velocity_update.lua")
end
if nn.SpatialFiniteElements == nil then
  dofile("spatial_finite_elements.lua")
end
if nn.VolumetricFiniteElements == nil then
  dofile("volumetric_finite_elements.lua")
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

function test.VelocityUpdateForward()
  local bsize = torch.random(1, 3)
  local height = torch.random(16, 32)
  local width = torch.random(16, 32)

  for matchManta = 0, 1 do
    for twoDim = 1, 0, -1 do
      local depth
      if twoDim == 1 then
        depth = 1
      else
        depth = torch.random(16, 32)
      end

      local p = torch.rand(bsize, 1, depth, height, width)
      local geom = torch.rand(bsize, 1, depth, height, width):gt(0.8):double()

      local mod = nn.VelocityUpdate(matchManta == 1)
      local deltaU = mod:forward({p, geom})

      -- Now use our ground truth lua function (which has been tested against
      -- the actual manta data in
      -- CNNFluids/torch/utils/test_calc_velocity_update.lua.
      local deltaUGT = torch.Tensor()
      if twoDim == 1 then
        deltaUGT:resize(bsize, 2, depth, height, width)
      else
        deltaUGT:resize(bsize, 3, depth, height, width)
      end
      for b = 1, bsize do
        torch.calcVelocityUpdate(deltaUGT[b], p[{b, 1}], geom[{b, 1}],
                                 matchManta == 1)
      end

      local err = deltaUGT - deltaU
      mytester:assertlt(err:abs():max(), precision,
                        'fprop error: matchManta ' .. matchManta ..
                        ', twoDim ' .. twoDim)
    end
  end
end

function test.VelocityUpdateBackward()
  local bsize = torch.random(1, 2)
  local height = torch.random(8, 12)
  local width = torch.random(8, 12)

  for matchManta = 0, 1 do
    for twoDim = 1, 0, -1 do
      local depth
      if twoDim == 1 then
        depth = 1
      else
        depth = torch.random(8, 12)
      end
      local p = torch.rand(bsize, 1, depth, height, width)
      local geom = torch.rand(bsize, 1, depth, height, width):gt(0.8):double()
      local mod = nn.Sequential()
      -- InjectTensor is a custom module just for this test. It takes in 'p'
      -- and a static 'geom' and simply outputs {p, geom}. It is so that the
      -- Jacobian tester only tests the pressure input gradient.
      mod:add(nn.InjectTensor(geom))
      mod:add(nn.VelocityUpdate(matchManta == 1))

      -- Test BPROP.
      err = jac.testJacobian(mod, p)
      mytester:assertlt(math.abs(err), precision,
                        'bprop error: matchManta ' .. matchManta ..
                        ', twoDim ' .. twoDim)
    end
  end
end

function test.VelocityUpdateCompareFEM()
  -- The no geometry case should be exactly the same as SpatialFiniteElements
  -- and VolumetricFiniteElements (when matchManta == false).
  -- This makes sure that we AT LEAST haven't completely destroyed our
  -- old legacy model.
  local bsize = torch.random(1, 2)
  local height = torch.random(8, 12)
  local width = torch.random(8, 12)
  for twoDim = 1, 0, -1 do
    local depth
    if twoDim == 1 then
      depth = 1
    else
      depth = torch.random(16, 32)
    end

    local p = torch.rand(bsize, 1, depth, height, width)
    local geom = torch.zeros(bsize, 1, depth, height, width)
    local matchManta = false
    local mod = nn.VelocityUpdate(matchManta)
    local deltaU = mod:forward({p, geom})
    local gradOutput = torch.rand(unpack(deltaU:size():totable()))
    local gradP = mod:backward({p, geom}, gradOutput)

    mod = nn.Sequential()
    if twoDim == 1 then
      mod:add(nn.Select(2, 1))
      mod:add(nn.SpatialFiniteElements(1, 1))
      -- The output will be size bsize x 1 x 2 x height x width, we need
      -- bsize x 2 x 1 x height x width.
      mod:add(nn.View(bsize, 2, 1, height, width))
    else
      mod:add(nn.VolumetricFiniteElements(1, 1, 1))
      -- The output will be size bsize x 1 x 3 x depth x height x width, we
      -- need bsize x 3 x depth x height x width.
      mod:add(nn.Select(2, 1, 1))
    end
    local deltaUFEM = mod:forward(p)
    local gradPFEM = mod:backward(p, gradOutput)

    local err = deltaU - deltaUFEM
    mytester:assertlt(err:abs():max(), precision,
                      'FPROP FEM does not match, twoDim ' .. twoDim)
    err = gradP[1] - gradPFEM
    mytester:assertlt(err:abs():max(), precision,
                      'BPROP FEM does not match, twoDim ' .. twoDim)
  end
end

-- NOTE: CPU <--> GPU comparison test can be found in torch/tfluids/test.lua.

-- Now run the test above
mytester:add(test)
mytester:run()
