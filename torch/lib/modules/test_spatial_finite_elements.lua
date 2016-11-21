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

-- A simple test for nn.SpatialFiniteElements.

local nn = require('nn')

if nn.SpatialFiniteElements == nil then
  dofile("spatial_finite_elements.lua")
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local fePrecision = 1e-4
local fePrecisionBorder = 1e-2  -- Border uses single-sided FE.
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

-- Implements a quadratic function of f = 0.5 * ((u * scale)^2 + (v * scale)^2)
-- and returns the closed form derivative.
local function createInput(scale, width, height, uvMin, uvMax)
  uvMin = uvMin or -1
  uvMax = uvMax or 1
  local vStep = (uvMax - uvMin) / (height - 1)
  local uStep = (uvMax - uvMin) / (width - 1)
  local v = torch.range(uvMin, uvMax, vStep)
  local u = torch.range(uvMin, uvMax, uStep)
  local v = v:resize(height, 1):repeatTensor(1, width)
  local u = u:resize(1, width):repeatTensor(height, 1)
  
  local vSq = v:clone():mul(scale):pow(2)  -- (v * scale) ^ 2
  local uSq = u:clone():mul(scale):pow(2)  -- (u * scale) ^ 2
  
  local input = vSq:add(uSq):mul(0.5)  -- f as described above.
  local dfdx = u:mul(scale * scale)
  local dfdy = v:mul(scale * scale)

  return input, dfdx, dfdy, uStep, vStep
end

function test.SpatialFiniteElementsKnown()
  local nbatch = 1
  local f = 1
  local h = 5
  local w = 5
  local scale = 1
  local input, dfdx, dfdy, stepX, stepY = createInput(scale, w, h)
  -- This results in the tensor:
  --   1.0000  0.6250  0.5000  0.6250  1.0000
  --   0.6250  0.2500  0.1250  0.2500  0.6250
  --   0.5000  0.1250  0.0000  0.1250  0.5000
  --   0.6250  0.2500  0.1250  0.2500  0.6250
  --   1.0000  0.6250  0.5000  0.6250  1.0000
  -- Putting this into Matlab, and calling gradient(input) on it:
  local dfdyM = torch.Tensor({{-0.3750, -0.3750, -0.3750, -0.3750, -0.3750},
                              {-0.2500, -0.2500, -0.2500, -0.2500, -0.2500},
                              {      0,       0,       0,       0,       0},
                              { 0.2500,  0.2500,  0.2500,  0.2500,  0.2500},
                              { 0.3750,  0.3750,  0.3750,  0.3750,  0.3750}})
  local dfdxM = torch.Tensor({{-0.3750, -0.2500,       0,  0.2500,  0.3750},
                              {-0.3750, -0.2500,       0,  0.2500,  0.3750},
                              {-0.3750, -0.2500,       0,  0.2500,  0.3750},
                              {-0.3750, -0.2500,       0,  0.2500,  0.3750},
                              {-0.3750, -0.2500,       0,  0.2500,  0.3750}})
  dfdyM:mul(1 / stepY)
  dfdxM:mul(1 / stepX)

  -- NOTE: the Matlab FEM approximation is ALSO wrong on the boundry, but it
  -- should be exactly the same as ours since they also use single-sided approx.

  local mod = nn.SpatialFiniteElements(stepX, stepY)
  local output = mod:forward(input:view(nbatch, f, h, w))
  
  local errHoriz = output[{{}, {}, 1, {}, {}}] - dfdxM
  local errVert = output[{{}, {}, 2, {}, {}}] - dfdyM

  mytester:assertlt(errHoriz:abs():max(), fePrecision, 'fprop error')
  mytester:assertlt(errVert:abs():max(), fePrecision, 'fprop error')
end

function test.SpatialFiniteElements()
  local nbatch = torch.random(1, 2)
  local f = torch.random(1, 3)
  local h = torch.random(30, 41)  -- Needs to be sufficiently large for FE.
  local w = torch.random(30, 41)

  -- We're going to define 'f' slices, with each slice being a quadratic
  -- in (u, v) with different scale. This will be an easy to calculate known
  -- derivative.
  local input = torch.Tensor(nbatch, f, h, w)
  local dfdx = torch.Tensor(nbatch, f, 1, h, w)
  local dfdy = torch.Tensor(nbatch, f, 1, h, w)
  local scales = torch.Tensor(nbatch, f)
  local xStep, yStep
  for b = 1, nbatch do
    for j = 1, f do
      -- Pick a random scale in [0.8, 1.2].
      scales[{b, j}] = torch.rand(1)[1] * 0.4 + 0.8
      local curInput, curDfdx, curDfdy, uStep, vStep =
          createInput(scales[{b, j}], w, h, -0.1, 0.1)  -- Decrease range.
      if xStep == nil then
        -- Record the input step size.
        xStep = uStep
      else
        -- All input slices should have the same step size.
        assert(xStep == uStep)
      end
      if yStep == nil then
        yStep = vStep
      else
        assert(yStep == vStep)
      end
      input[{b, j, {}, {}}]:copy(curInput)
      dfdx[{b, j, {}, {}}]:copy(curDfdx)
      dfdy[{b, j, {}, {}}]:copy(curDfdy)
    end
  end
  local grad = torch.cat(dfdx, dfdy, 3)

  -- Construct an instance of the module.
  local mod = nn.SpatialFiniteElements(xStep, yStep)

  -- Test FPROP.
  local output = mod:forward(input)
  assert(output:dim() == 5 and output:size(1) == nbatch and
         output:size(2) == f and output:size(3) == 2 and output:size(4) == h and
         output:size(5) == w)
  local err = (grad - output)
  local errBorder = err:clone()
  errBorder[{{}, {}, {}, {2, -2}, {2, -2}}]:fill(0)
  mytester:assertlt(errBorder:abs():max(), fePrecisionBorder, 'fprop error')
  local errCenter = err[{{}, {}, {}, {2, -2}, {2, -2}}]:clone()
  mytester:assertlt(errCenter:abs():max(), fePrecision, 'fprop error')

  -- Test BPROP.
  err = jac.testJacobian(mod, input)
  mytester:assertlt(err, precision, 'bprop error ')
end



-- Now run the test above
mytester:add(test)
mytester:run()
