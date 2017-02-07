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

-- This module allows us to inject a static tensor input.
-- It is used for testing nn.InjectTensor.
--
-- The input is a tensor.
-- The output is a table of {input, self.staticData}. if staticData is a
-- tensor or {input, self.staticData[1], ..., self.staticData[end]} if it is
-- a table.

local InjectTensor, parent =
  torch.class('nn.InjectTensor', 'nn.Module')

function InjectTensor:__init(staticData)
  parent.__init(self)
  if torch.isTensor(staticData) then
    self.staticData = {staticData}
  else
    assert(torch.type(staticData) == 'table')
    self.staticData = staticData
    for i = 1, #staticData do
      assert(torch.isTensor(staticData[i]))
    end
  end
  self.output = {}
end

function InjectTensor:updateOutput(input)
  assert(torch.isTensor(input))
  self.output[1] = input
  for i = 1, #self.staticData do
    self.output[i + 1] = self.staticData[i]
  end

  return self.output
end

function InjectTensor:updateGradInput(input, gradOutput)
  -- Assume updateOutput has already been called.
  assert(torch.type(gradOutput) == 'table' and
         #gradOutput == #self.staticData + 1)
  assert(gradOutput[1]:isSameSizeAs(input))
  self.gradInput = gradOutput[1]
  
  return self.gradInput
end

function InjectTensor:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function InjectTensor:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function InjectTensor:type(type)
  parent.type(self, type)
  for i = 1, #self.staticData do
    self.staticData[i] = self.staticData[i]:type(type)
  end
  self.output = {}
  return self
end

function InjectTensor:clearState()
  parent.clearState(self)
  self.output = {}
  return self
end

