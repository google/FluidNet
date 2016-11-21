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

-- Jonathan Tompson
-- See the file "LICENSE" for the full license governing this code.

require 'nn'
require 'cunn'

dofile('fluid_l1_criterion.lua')
dofile('fluid_l2_criterion.lua')
dofile('velocity_divergence_criterion.lua')
dofile('partial_derivative.lua')

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = {}

local function criterionJacobianTest1D(cri, input, target)
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end

local function criterionJacobianTest1DTable(cri, input0, target)
   -- supposes input is a tensor, which is splitted in the first dimension
   local input = input0:split(1,1)
   for i=1,#input do
      input[i] = input[i][1]
   end
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(input0)
   local input_s = input0:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input0:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end
   local centraldiff_dfdx_t = centraldiff_dfdx:split(1,1)
   for i=1,#centraldiff_dfdx_t do
      centraldiff_dfdx_t[i] = centraldiff_dfdx_t[i][1]
   end
   for i=1,#centraldiff_dfdx_t do
      -- compare centraldiff_dfdx with :backward()
      local err = (centraldiff_dfdx_t[i] - dfdx[i]):abs():max()
      mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
   end
end

function test.FluidL1Criterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = nn.FluidL1Criterion()
   criterionJacobianTest1D(cri, input, target)
end

function test.FluidL2Criterion()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = nn.FluidL2Criterion()
   criterionJacobianTest1D(cri, input, target)
end

function test.PartialDerivative()
   local input = torch.rand(10)
   local target = input:clone():add(torch.rand(10))
   local cri = nn.PartialDerivative(1,1)
   criterionJacobianTest1D(cri, input, target)
end

-- Now run the test above
mytester:add(test)
mytester:run()
