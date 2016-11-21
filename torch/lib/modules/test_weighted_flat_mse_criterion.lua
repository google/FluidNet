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

require 'nn'
if nn.WeightedFlatMSECriterion == nil then
  dofile('weighted_flat_mse_criterion.lua')
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework.
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

local function criterionJacobianTest(
    cri, input, target, weight, altTarget, altWeight)
  local eps = 1e-6
  cri:forward(input, target, weight, altTarget, altWeight)
  local dfdx = cri:backward(input, target, weight, altTarget, altWeight)
  -- for each input perturbation, do central difference
  local centralDiffDfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx)
  local pinput = input:data()
  local pcentralDiffDfdx = centralDiffDfdx:data()
  for i = 0, input:nElement() - 1 do
    -- f(xi + h).
    pinput[i] = pinput[i] + eps
    local fx1 = cri:forward(input, target, weight, altTarget, altWeight)
    -- f(xi - h).
    pinput[i] = pinput[i] - 2 * eps
    local fx2 = cri:forward(input, target, weight, altTarget, altWeight)
    -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h.
    local cdfx = (fx1 - fx2) / (2 * eps)
    -- store f' in appropriate place.
    pcentralDiffDfdx[i] = cdfx
    -- reset pinput[i].
    pinput[i] = pinput[i] + eps
  end

  -- compare centralDiffDfdx with :backward()
  local err = (centralDiffDfdx - dfdx):abs():max()
  return err
end

function test.WeightedFlatMSECriterion()
  for dim = 2, 4 do
    for sizeAverage = 0, 1 do
      -- Create a random input and target size.
      local sz = {}
      for i = 1, dim do
        sz[i] = torch.random(3, 7)
      end

      local input = torch.rand(unpack(sz))
      local target = torch.rand(unpack(sz))
      local weight = torch.rand(unpack(sz))

      local criterion = nn.WeightedFlatMSECriterion()
      criterion.sizeAverage = sizeAverage == 1
      local critVal = criterion:forward(input, target, weight)

      local err = criterionJacobianTest(criterion, input,
                                        target, weight)
      mytester:assertlt(err, precision, 'error on FEM: dim ' .. dim ..
                                        ' sizeAverage ' .. sizeAverage)
    end
  end
end

-- Now run the test above.
mytester:add(test)
mytester:run()
