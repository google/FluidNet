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

-- A simple test for nn.VolumetricFiniteElements.

local nn = require('nn')

if nn.VolumetricFiniteElements == nil then
  dofile("volumetric_finite_elements.lua")
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

-- Implements a quadratic function of f = 0.5 * ((u * scale)^2 + (v * scale)^2 +
-- (z * scale)^2) and returns the closed form derivative.
function createInput(scale, width, height, depth)
  local min = -1
  local max = 1
  local dStep = (max - min) / (depth - 1)
  local vStep = (max - min) / (height - 1)
  local uStep = (max - min) / (width - 1)
  local d = torch.range(min, max, dStep)
  local v = torch.range(min, max, vStep)
  local u = torch.range(min, max, uStep)
  local d = d:resize(depth, 1, 1):repeatTensor(1, height, width)
  local v = v:resize(1, height, 1):repeatTensor(depth, 1, width)
  local u = u:resize(1, 1, width):repeatTensor(depth, height, 1)
 
  local dSq = d:clone():mul(scale):pow(2)  -- (d * scale) ^ 2
  local vSq = v:clone():mul(scale):pow(2)  -- (v * scale) ^ 2
  local uSq = u:clone():mul(scale):pow(2)  -- (u * scale) ^ 2
  
  local input = dSq:add(vSq):add(uSq):mul(0.5)  -- f as described above.
  local dfdx = u:mul(scale * scale)
  local dfdy = v:mul(scale * scale)
  local dfdz = d:mul(scale * scale)

  return input, dfdx, dfdy, dfdz, uStep, vStep, dStep
end

function test.VolumetricFiniteElementsKnown()
  local nbatch = 1
  local f = 1
  local d = 5
  local h = 5
  local w = 5
  local scale = 1
  local input, dfdx, dfdy, dfdz, stepX, stepY, stepZ =
      createInput(scale, w, h, d)
  -- This results in the tensor:
  -- (1,.,.) = 
  --   1.5000  1.1250  1.0000  1.1250  1.5000
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  --   1.0000  0.6250  0.5000  0.6250  1.0000
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  --   1.5000  1.1250  1.0000  1.1250  1.5000
  -- 
  -- (2,.,.) = 
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  --   0.7500  0.3750  0.2500  0.3750  0.7500
  --   0.6250  0.2500  0.1250  0.2500  0.6250
  --   0.7500  0.3750  0.2500  0.3750  0.7500
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  -- 
  -- (3,.,.) = 
  --   1.0000  0.6250  0.5000  0.6250  1.0000
  --   0.6250  0.2500  0.1250  0.2500  0.6250
  --   0.5000  0.1250  0.0000  0.1250  0.5000
  --   0.6250  0.2500  0.1250  0.2500  0.6250
  --   1.0000  0.6250  0.5000  0.6250  1.0000
  -- 
  -- (4,.,.) = 
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  --   0.7500  0.3750  0.2500  0.3750  0.7500
  --   0.6250  0.2500  0.1250  0.2500  0.6250
  --   0.7500  0.3750  0.2500  0.3750  0.7500
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  -- 
  -- (5,.,.) = 
  --   1.5000  1.1250  1.0000  1.1250  1.5000
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  --   1.0000  0.6250  0.5000  0.6250  1.0000
  --   1.1250  0.7500  0.6250  0.7500  1.1250
  --   1.5000  1.1250  1.0000  1.1250  1.5000
  -- Putting this into Matlab, and calling gradient(input) on it:
  local dfdxM = torch.Tensor({-0.3750, -0.2500, 0, 0.2500, 0.3750})
  dfdxM = dfdxM:view(1, 1, 5):repeatTensor(5, 5, 1):contiguous()
  local dfdyM = torch.Tensor({-0.3750, -0.2500, 0, 0.2500, 0.3750})
  dfdyM = dfdyM:view(1, 5, 1):repeatTensor(5, 1, 5):contiguous()
  local dfdzM = torch.Tensor({-0.3750, -0.2500, 0, 0.2500, 0.3750})
  dfdzM = dfdzM:view(5, 1, 1):repeatTensor(1, 5, 5):contiguous()
  dfdzM:mul(1 / stepZ)
  dfdyM:mul(1 / stepY)
  dfdxM:mul(1 / stepX)

  -- NOTE: the Matlab FEM approximation is ALSO wrong on the boundry, but it
  -- should be exactly the same as ours since they also use single-sided approx.

  local mod = nn.VolumetricFiniteElements(stepX, stepY, stepZ)
  local output = mod:forward(input:view(nbatch, f, d, h, w))
  
  local errHoriz = output[{{}, {}, 1, {}, {}, {}}] - dfdxM
  local errVert = output[{{}, {}, 2, {}, {}, {}}] - dfdyM
  local errDepth = output[{{}, {}, 3, {}, {}, {}}] - dfdzM

  mytester:assertlt(errHoriz:abs():max(), fePrecision, 'fprop error')
  mytester:assertlt(errVert:abs():max(), fePrecision, 'fprop error')
  mytester:assertlt(errDepth:abs():max(), fePrecision, 'fprop error')
end

function test.VolumetricFiniteElements()
  local nbatch = torch.random(1, 2)
  local f = torch.random(1, 3)
  local d = torch.random(4, 6)
  local h = torch.random(4, 6)
  local w = torch.random(4, 6)
  local stepX = torch.rand(1)[1] + 1
  local stepY = torch.rand(1)[1] + 1
  local stepZ = torch.rand(1)[1] + 1

  local input = torch.Tensor(nbatch, f, d, h, w)
  local mod = nn.VolumetricFiniteElements(stepX, stepY, stepZ)

  -- Test BPROP.
  err = jac.testJacobian(mod, input)
  mytester:assertlt(err, precision, 'bprop error ')
end

-- Now run the test above
mytester:add(test)
mytester:run()
