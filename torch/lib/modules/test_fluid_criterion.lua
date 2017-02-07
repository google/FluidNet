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
if nn.WeightedFlatMSECriterion == nil then
  dofile('weighted_flat_mse_criterion.lua')
end
if nn.FluidCriterion == nil then
  dofile('fluid_criterion.lua')
end
if nn.MSESICriterion == nil then
  dofile('mse_si_criterion.lua')
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework.
local eps = 1e-5
local precision = eps * 2
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

-- This is a little tricky since the input to cri is a table of tensors.
-- To simplify the FEM, we'll pass in an aux network to split the input for
-- us.
local function criterionJacobianTest(cri, splitNet, inputCon, target)
  local input = splitNet:forward(inputCon)
  cri:forward(input, target)
  local dfdx = cri:backward(input, target)
  local dfdxCon = splitNet:backward(inputCon, dfdx)
  -- for each input perturbation, do central difference
  local centralDiffDfdx = torch.Tensor():typeAs(dfdxCon):resizeAs(dfdxCon)
  local pinput = inputCon:data()
  local pcentralDiffDfdx = centralDiffDfdx:data()
  for i = 0, inputCon:nElement() - 1 do
    -- f(xi + h).
    pinput[i] = pinput[i] + eps
    input = splitNet:forward(inputCon)
    local fx1 = cri:forward(input, target)
    -- f(xi - h).
    pinput[i] = pinput[i] - 2 * eps
    input = splitNet:forward(inputCon)
    local fx2 = cri:forward(input, target)
    -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h.
    local cdfx = (fx1 - fx2) / (2 * eps)
    -- store f' in appropriate place.
    pcentralDiffDfdx[i] = cdfx
    -- reset pinput[i].
    pinput[i] = pinput[i] + eps
  end

  -- compare centralDiffDfdx with :backward()
  local err = (centralDiffDfdx - dfdxCon):abs():max()
  return err
end

local function localNumbersToString()
  local i = 1
  local str = ''
  repeat
    local k, v = debug.getlocal(2, i)
    if k then
      if type(v) == 'number' then
        str = str .. k .. '=' .. v .. ' '
      end
      i = i + 1
    end
  until nil == k
  return str
end

function test.FluidCriterion()
  local batchSz = torch.random(1, 2)
  local d = torch.random(6, 10)
  local h = torch.random(6, 10)
  local w = torch.random(6, 10)

  for withBorderWeight = 0, 1 do
    for sizeAverage = 0, 1 do
      for is3D = 0, 1 do
        for addP = 0, 1 do
          for addU = 0, 1 do
            for addDiv = 0, 1 do
              local numUChan = 3
              local curD = d
              if is3D == 0 then
                numUChan = 2
                curD = 1
              end

              local target = {
                  torch.rand(batchSz, 1, curD, h, w):mul(2):add(-1),  -- p
                  torch.rand(
                      batchSz, numUChan, curD, h, w):mul(2):add(-1),  -- U
                  torch.rand(batchSz, 1, curD, h, w):gt(0.9):double()  -- flags
              }

              -- Make sure at least one of the cells is occupied.
              assert(target[3]:sum() > 0)

              -- Flags is an occupancy grid. We need it to be a true
              -- flag grid.
              target[3] = target[3] * tfluids.CellType.TypeObstacle +
                  (1 - target[3]) * tfluids.CellType.TypeFluid
   
              local pLambda = torch.rand(1):add(1)[1]  -- in [1, 2]
              local uLambda = torch.rand(1):add(1)[1]
              local divLambda = torch.rand(1):add(1)[1]
              local borderWeight, borderWidth
              if withBorderWeight == 1 then
                borderWeight = torch.rand(1):add(2)[1]  -- in [2, 3]
                borderWidth = torch.random(2, 5)
              end
   
              pLambda = pLambda * addP
              uLambda = uLambda * addU
              divLambda = divLambda * addDiv

              local crit = nn.FluidCriterion(pLambda, uLambda, divLambda,
                                             borderWeight, borderWidth)
              crit.sizeAverage = sizeAverage == 1
              local splitNet = nn.ConcatTable()
              splitNet:add(nn.Narrow(2, 1, 1))  -- pPred
              splitNet:add(nn.Narrow(2, 2, numUChan))  -- UPred

              local input =
                  torch.rand(batchSz, numUChan + 1, curD, h, w):mul(2):add(-1)
   
              local err = criterionJacobianTest(crit, splitNet, input,
                                                target)

              mytester:assertlt(err, precision,
                                'error on FEM: ' .. localNumbersToString())
            end
          end
        end
      end
    end
  end
end

-- Now run the test above.
mytester:add(test)
mytester:run()
