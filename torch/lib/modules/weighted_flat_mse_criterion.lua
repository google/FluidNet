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

-- The error function for this module is simply:
--   err = sum_i (||weight_i * (input_i - target_i)||^2)
-- i.e. the weight tensor is the same size as input (and target) and is applied
-- 'flat' to the inputs.
--
-- Note that WeightedMSECriterion in nn is DEFINITELY not what you would expect
-- and does not implement the above objective.

local nn = require('nn')

local WeightedFlatMSECriterion, parent =
    torch.class('nn.WeightedFlatMSECriterion', 'nn.Criterion')

function WeightedFlatMSECriterion:__init()
  parent.__init(self)
  self.sizeAverage = true
  self._inputBuffer = torch.Tensor()
  self._targetBuffer = torch.Tensor()
end

function WeightedFlatMSECriterion:updateOutput(input, target, weight)
  assert(input:isSameSizeAs(target), 'input and target size mismatch')
  assert(input:isSameSizeAs(weight), 'input and target size mismatch')
  self._inputBuffer:resizeAs(input)
  self._targetBuffer:resizeAs(target)
  self._inputBuffer:copy(input):cmul(weight)
  self._targetBuffer:copy(target):cmul(weight)

  -- Thanks to FB, the torch codebase is now fractured into old style C
  -- calls and new style. We need to be able to support both.
  if self._inputBuffer.nn == nil or
      self._inputBuffer.nn.MSECriterion_updateOutput == nil then
    self._outputTensor = self._outputTensor or input.new(1)
    self._inputBuffer.THNN.MSECriterion_updateOutput(
        self._inputBuffer:cdata(), self._targetBuffer:cdata(),
        self._outputTensor:cdata(), self.sizeAverage)
    self.output = self._outputTensor[1]
  else
    self.output =
        self._inputBuffer.nn.MSECriterion_updateOutput(self, self._inputBuffer,
                                                       self._targetBuffer)
  end

  return self.output
end

function WeightedFlatMSECriterion:updateGradInput(input, target, weight)
  -- Assume updateOutput has already been called (and weight has been applied).
  self.gradInput:resizeAs(input)
  if self._inputBuffer.nn == nil or
      self._inputBuffer.nn.MSECriterion_updateGradInput == nil then
    self._inputBuffer.THNN.MSECriterion_updateGradInput(
        self._inputBuffer:cdata(), self._targetBuffer:cdata(),
        self.gradInput:cdata(), self.sizeAverage)
  else
    self.gradInput = self._inputBuffer.nn.MSECriterion_updateGradInput(
        self, self._inputBuffer, self._targetBuffer)
  end

  self.gradInput:cmul(weight)
  return self.gradInput
end

-- Overload the forward, backward and __call__ functions to inject the weight
-- tensor.
function WeightedFlatMSECriterion:forward(input, target, weight)
  return self:updateOutput(input, target, weight)
end

function WeightedFlatMSECriterion:backward(input, target, weight)
  return self:updateGradInput(input, target, weight)
end

function WeightedFlatMSECriterion:__call__(input, target, weight)
  self.output = self:forward(input, target, weight)
  self.gradInput = self:backward(input, target, weight)
  return self.output, self.gradInput
end
