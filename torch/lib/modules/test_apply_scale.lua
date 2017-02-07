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

if nn.ApplyScale == nil then
  dofile("apply_scale.lua")
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-3
local mytester = torch.Tester()
local jac = nn.Jacobian
local tests = torch.TestSuite()

function tests.ApplyScale()
  for nDims = 2, 5 do
    for invertScale = 0, 1 do
      local inputSize = {}
      for i = 1, nDims do
        inputSize[i] = torch.random(1, 3)
      end
      local batchSize = inputSize[1]

      local input = torch.rand(unpack(inputSize))
      local scale = torch.rand(batchSize, 1)
      
      local mod = nn.ApplyScale(invertScale == 1)
      
      -- Test FPROP.
      local output = mod:forward({input, scale})
     
      local outputGT = input:clone()
      for i = 1, batchSize do
        if invertScale == 1 then
          outputGT[i]:div(scale[{i, 1}])
        else
          outputGT[i]:mul(scale[{i, 1}])
        end
      end
      mytester:assertlt((output - outputGT):abs():max(), precision)

      -- Test BPROP.
      -- We have to construct a network that takes in a tensor and splits
      -- it into the input and scale values.
      local inputNumel = 1
      for i = 2, #inputSize do
        inputNumel = inputNumel * inputSize[i]
      end
      local scaleNumel = 1
      
      local input = torch.rand(batchSize, inputNumel + scaleNumel)

      local net = nn.Sequential()
      
      local par = nn.ConcatTable()
      local inputSelect = nn.Sequential()
      inputSelect:add(nn.Narrow(2, 1, inputNumel))
      inputSelect:add(nn.View(unpack(inputSize)))
      local scaleSelect = nn.Sequential()
      scaleSelect:add(nn.Narrow(2, inputNumel + 1, scaleNumel))
      scaleSelect:add(nn.View(batchSize, 1))
      par:add(inputSelect)
      par:add(scaleSelect)

      net:add(par)
      net:add(mod)

      local err = jac.testJacobian(net, input)
      mytester:assertlt(err, precision, 'bprop error ')
    end
  end
end

-- Now run the test above
mytester:add(tests)
mytester:run()
