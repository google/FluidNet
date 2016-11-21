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

require 'cunn'
if nn.VolumetricConvolutionUpsample == nil then
  dofile('volumetric_convolution_upsample.lua')
end

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework.
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = torch.TestSuite()

function test.VolumetricConvolutionUpsample()
  local batchSize = torch.random(1, 2)
  local nInputPlane = torch.random(1, 5)
  local nOutputPlane = torch.random(1, 5)
  local kT = torch.random(1, 4)
  local kW = torch.random(1, 4)
  local kH = torch.random(1, 4)
  local dT = torch.random(1, 5)
  local dW = torch.random(1, 5)
  local dH = torch.random(1, 5)
  local padT = torch.random(0, 5)
  local padW = torch.random(0, 5)
  local padH = torch.random(0, 5)
  local scaleT = torch.random(1, 3)
  local scaleW = torch.random(1, 3)
  local scaleH = torch.random(1, 3)
  local widthIn = torch.random(5, 8)
  local heightIn = torch.random(5, 8)
  local depthIn = torch.random(5, 8)

  local module = nn.VolumetricConvolutionUpsample(
     nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH,
     scaleT, scaleW, scaleH)

  -- For gradBias/Weight tests we'll want to test the sub-module params.
  -- Technically, no one else should ever directly touch these params, but
  -- we need to for testing.
  local conv = module.modules[1]
  assert(torch.type(conv) == 'nn.VolumetricConvolution')

  local input = torch.rand(batchSize, nInputPlane, depthIn, heightIn, widthIn)
  local output = module:forward(input)
  -- Output size explained at:
  -- github.com/torch/nn/blob/master/doc/convolution.md#nn.VolumetricConvolution
  local oDepthConv = math.floor((depthIn + 2 * padT - kT) / dT + 1)
  assert(oDepthConv == conv.output:size(3))
  local oHeightConv = math.floor((heightIn + 2 * padH - kH) / dH + 1)
  assert(oHeightConv == conv.output:size(4))
  local oWidthConv = math.floor((widthIn  + 2 * padW - kW) / dW + 1)
  assert(oWidthConv == conv.output:size(5))
  assert(oWidthConv * scaleW == output:size(5))
  assert(oHeightConv * scaleH == output:size(4))
  assert(oDepthConv * scaleT == output:size(3))

  -- Check BPROP is correct.
  local err = jac.testJacobian(module, input)
  mytester:assertlt(err, precision, 'error on bprop')

  local err = jac.testJacobianParameters(module, input, conv.weight,
                                         conv.gradWeight)
  mytester:assertlt(err , precision, 'error on weight ')

  local err = jac.testJacobianParameters(module, input,
                                         conv.bias, conv.gradBias)
  mytester:assertlt(err , precision, 'error on bias ')

  local err = jac.testJacobianUpdateParameters(module, input, conv.weight)
  mytester:assertlt(err , precision, 'error on weight [direct update] ')

  local err = jac.testJacobianUpdateParameters(module, input, conv.bias)
  mytester:assertlt(err , precision, 'error on bias [direct update] ')

  local ferr, berr = jac.testIO(module, input)
  mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
  mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

  module:zeroGradParameters()
  local outputCPU = module:forward(input):clone()
  local gradOutput = torch.rand(unpack(output:size():totable()))
  local gradInputCPU = module:backward(input, gradOutput):clone()
  local paramsCPU, gradParamsCPU = module:parameters()

  -- Convert to CUDA and make sure we get the same gradients.
  module:cuda()
  module:zeroGradParameters()
  local outputGPU = module:forward(input:cuda())
  local gradInputGPU = module:backward(input:cuda(), gradOutput:cuda())
  local paramsGPU, gradParamsGPU = module:parameters()

  local err = (outputCPU - outputGPU:double()):abs():max()
  mytester:assertlt(err , precision, 'output GPU error')

  local err = (gradInputCPU - gradInputGPU:double()):abs():max()
  mytester:assertlt(err , precision, 'gradInput GPU error')

  assert(#paramsCPU == #paramsGPU and #gradParamsCPU == #gradParamsGPU)
  assert(#paramsCPU == #gradParamsCPU)
  for i = 1, #paramsCPU do
    local err = (paramsCPU[i] - paramsGPU[i]:double()):abs():max()
    mytester:assertlt(err , precision, 'params GPU error')

    local err = (gradParamsCPU[i] - gradParamsGPU[i]:double()):abs():max()
    mytester:assertlt(err , precision * 10, 'gradParams GPU error')
  end
end

-- Now run the test above.
mytester:add(test)
mytester:run()

