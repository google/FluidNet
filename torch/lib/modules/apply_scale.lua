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

-- This method simply takes in a tensor and a scale tensor and applies a PER
-- batch sample scale value to the input. If invertScale == true then we
-- will apply 1/scale.
--
-- i.e. output(ibatch, ...) = input(ibatch, ...) * scale(ibatch, 1)
-- or
-- output(ibatch, ...) = input(ibatch, ...) / scale(ibatch, 1)

local ApplyScale, parent = torch.class('nn.ApplyScale', 'nn.Module')

function ApplyScale:__init(invertScale)
  parent.__init(self)
  if invertScale then
    self.modules = {nn.CDivTable()}
  else
    self.modules = {nn.CMulTable()}
  end
  self.output = self.modules[1].output
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

local function scaleExpand(input, scale)
  assert(torch.isTensor(input) and torch.isTensor(scale))
  assert(input:size(1) == scale:size(1))  -- Batch size must match.
  assert(scale:dim() == 2 and scale:size(2) == 1)

  -- Add unary dims to scale.
  local sz = torch.ones(input:dim()):totable()
  sz[1] = input:size(1)
  scale = scale:view(unpack(sz))
  scale = scale:expandAs(input)
  return scale
end

function ApplyScale:updateOutput(input)
  assert(torch.type(input) == 'table' and #input == 2)
  local scale = input[2]
  input = input[1]
  scale = scaleExpand(input, scale)

  self.output = self.modules[1]:updateOutput({input, scale})

  return self.output 
end

function ApplyScale:updateGradInput(input, gradOutput)
  assert(torch.type(input) == 'table' and #input == 2)
  local scale = input[2]
  input = input[1]

  assert(gradOutput:isSameSizeAs(input))
  self.gradInput[1]:resizeAs(input)
  self.gradInput[2]:resizeAs(scale)

  scale = scaleExpand(input, scale)

  local gradInput = self.modules[1]:updateGradInput({input, scale}, gradOutput)

  self.gradInput[1]:copy(gradInput[1])

  -- Now we need to accumulate the scale gradients to just a scalar value per
  -- batch (because the forward op is essentially a broadcast).
  local scaleGrad = gradInput[2]
  scaleGrad = scaleGrad:view(scaleGrad:size(1),
                             scaleGrad:numel() / scaleGrad:size(1))
  torch.sum(self.gradInput[2], scaleGrad, 2)

  return self.gradInput
end

function ApplyScale:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function ApplyScale:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function ApplyScale:type(type)
  self.modules[1]:type(type)
  self.output = self.modules[1].output
  self.gradInput[1] = self.gradInput[1]:type(type)
  self.gradInput[2] = self.gradInput[2]:type(type)
  return self
end

function ApplyScale:clearState()
  self.modules[1]:clearState()
  self.gradInput[1]:set()
  self.gradInput[2]:set()
  self.output:set()
  return self
end

