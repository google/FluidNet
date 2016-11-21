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

-- A simple test for nn.Variance.

local nn = require('nn')

if nn.Variance == nil then
  dofile("variance.lua")
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local tests = torch.TestSuite()

function tests.Variance()
  for nDims = 1, 4 do
    for varDim = 1, nDims do
      local inputSize = {}
      for i = 1, nDims do
        inputSize[i] = torch.random(5, 10)
      end
      local input = torch.rand(unpack(inputSize))
      
      local mod = nn.Variance(varDim)
      
      -- Test FPROP.
      local output = mod:forward(input)
      local outputGT = torch.var(input, varDim)
      mytester:assertlt((output - outputGT):abs():max(), precision)

      -- Test BPROP.
      local err = jac.testJacobian(mod, input)
      mytester:assertlt(err, precision, 'bprop error ')
    end
  end
end

-- Now run the test above
mytester:add(tests)
mytester:run()
