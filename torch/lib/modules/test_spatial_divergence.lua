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

-- A simple test for nn.SpatialDivergence.
-- NOTE: WE USE nn.SpatialFiniteElements TO CALCULATE GROUND-TRUTH SO WE
-- ASSUME THAT LAYER IS CORRECT.

local nn = require('nn')

if nn.SpatialFiniteElements == nil then
  dofile("spatial_finite_elements.lua")
end
if nn.SpatialDivergence == nil then
  dofile("spatial_divergence.lua")
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

function test.SpatialDivergence()
  local nbatch = torch.random(1, 3)
  local h = torch.random(30, 45)  -- Needs to be sufficiently large for FE.
  local w = torch.random(30, 45)

  -- Build a network that uses nn.SpatialFiniteElements to calculate the
  -- divergence.
  local xStep = torch.rand(1)[1] + 0.5  -- in [0.5, 1.5]
  local yStep = torch.rand(1)[1] + 0.5
  local modGT = nn.Sequential()
  modGT:add(nn.SpatialFiniteElements(xStep, yStep))
  -- Output of the above is nbatch x 2 x 2 x h x w.
  local par = nn.ConcatTable()
  local select_1_1 = nn.Sequential()
  select_1_1:add(nn.Narrow(2, 1, 1)):add(nn.Select(3, 1))
  local select_2_2 = nn.Sequential()
  select_2_2:add(nn.Narrow(2, 2, 1)):add(nn.Select(3, 2))
  par:add(select_1_1)
  par:add(select_2_2)
  modGT:add(par)
  -- The output now is a table of 2 tensors of size nbatch x 1 x h x w.
  modGT:add(nn.CAddTable())

  local input = torch.rand(nbatch, 2, h, w)
  local outputGT = modGT:updateOutput(input)
  local gradOutput = torch.rand(unpack(outputGT:size():totable()))
  local gradInputGT = modGT:updateGradInput(input, gradOutput)

  local mod = nn.SpatialDivergence(xStep, yStep)
  local output = mod:forward(input)
  local gradInput = mod:updateGradInput(input, gradOutput)

  mytester:assertlt((output - outputGT):abs():max(), precision)
  mytester:assertlt((gradInput - gradInputGT):abs():max(), precision)
end

-- Now run the test above
mytester:add(test)
mytester:run()
