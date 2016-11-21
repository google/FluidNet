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
if nn.VelocityDivergence == nil then
  dofile("velocity_divergence.lua")
end
if nn.InjectTensor == nil then
  dofile("inject_tensor.lua")
end
if nn.SpatialDivergence == nil then
  dofile("spatial_divergence.lua")
end
if nn.VolumetricDivergence == nil then
  dofile("volumetric_divergence.lua")
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

function test.VelocityDivergenceBackward()
  local bsize = torch.random(1, 2)
  local height = torch.random(8, 12)
  local width = torch.random(8, 12)

  for twoDim = 1, 0, -1 do
    local depth, nuchan
    if twoDim == 1 then
      depth = 1
      nuchan = 2
    else
      depth = torch.random(8, 12)
      nuchan = 3
    end
    local U = torch.rand(bsize, nuchan, depth, height, width)
    local geom = torch.rand(bsize, depth, height, width):gt(0.8):double()
    local mod = nn.Sequential()
    -- InjectTensor is a custom module just for this test. It takes in 'p'
    -- and a static 'geom' and simply outputs {p, geom}. It is so that the
    -- Jacobian tester only tests the pressure input gradient.
    mod:add(nn.InjectTensor(geom))
    mod:add(nn.VelocityDivergence())

    -- Test BPROP.
    err = jac.testJacobian(mod, U)
    mytester:assertlt(math.abs(err), precision,
                      'bprop error: twoDim ' .. twoDim)
  end
end

function test.VelocityDivergenceCompareFEM()
  -- The no geometry case should be exactly the same as SpatialDivergence
  -- and VolumetricDivergence.
  local bsize = torch.random(1, 2)
  local height = torch.random(8, 12)
  local width = torch.random(8, 12)
  for twoDim = 1, 0, -1 do
    local depth, nuchan
    if twoDim == 1 then
      depth = 1
      nuchan = 2
    else
      depth = torch.random(16, 32)
      nuchan = 3
    end

    local U = torch.rand(bsize, nuchan, depth, height, width)
    local geom = torch.zeros(bsize, depth, height, width)
    local mod = nn.VelocityDivergence()
    local UDiv = mod:forward({U, geom})
    local gradOutput = torch.rand(unpack(UDiv:size():totable()))
    local gradU = mod:backward({U, geom}, gradOutput)

    mod = nn.Sequential()
    if twoDim == 1 then
      mod:add(nn.Select(3, 1, 1))  -- Remove unary z dim.
      mod:add(nn.SpatialDivergence())
    else
      mod:add(nn.VolumetricDivergence())
      -- The output will be size bsize x 1 x depth x height x width, we
      -- need bsize x depth x height x width.
      mod:add(nn.Select(2, 1, 1))  -- Remove unary feat dim.
    end
    local UDivFEM = mod:forward(U)
    local gradUFEM = mod:backward(U, gradOutput)

    local err = UDiv - UDivFEM
    mytester:assertlt(err:abs():max(), precision,
                      'FPROP FEM does not match, twoDim ' .. twoDim)
    err = gradU[1] - gradUFEM
    mytester:assertlt(err:abs():max(), precision,
                      'BPROP FEM does not match, twoDim ' .. twoDim)
  end
end

-- NOTE: CPU <--> GPU comparison test can be found in torch/tfluids/test.lua.

-- Now run the test above
mytester:add(test)
mytester:run()
