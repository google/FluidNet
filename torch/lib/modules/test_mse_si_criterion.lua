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
if nn.MSESICriterion == nil then
  dofile('mse_si_criterion.lua')
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework.
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

local function criterionJacobianTest(cri, input, target)
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
  return err
end

function test.WeightedFlatMSECriterion()
  for numDimsNonBatch = 1, 4 do
    for sizeAverage = 0, 1 do
      -- Create a random input and target size.
      local sz = {}
      for i = 1, numDimsNonBatch + 1 do
        sz[i] = torch.random(3, 7)
      end

      local input = torch.rand(unpack(sz))
      local target = torch.rand(unpack(sz))

      local criterion = nn.MSESICriterion(numDimsNonBatch)
      criterion.sizeAverage = sizeAverage == 1
      local critVal = criterion:forward(input, target)

      local delta = input - target
      local n = input:numel()
      local sumSq = delta:clone()
      for i = numDimsNonBatch + 1, 2, -1 do
        sumSq = sumSq:sum(i)
      end
      sumSq = sumSq:pow(2)
      sumSq = sumSq:sum(1):squeeze()

      local critValManual = (1 / n) * delta:clone():pow(2):sum() +
          (1 / (n * n)) * sumSq

      local err = math.abs(critValManual - critVal)
      mytester:assertlt(err, precision, 'error on forward')

      err = criterionJacobianTest(criterion, input, target)
      mytester:assertlt(err, precision, 'error on FEM')
    end
  end
end

-- Now run the test above.
mytester:add(test)
mytester:run()
