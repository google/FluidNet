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

local VolumetricUpSamplingNearest, parent =
  torch.class('tfluids.VolumetricUpSamplingNearest', 'nn.Module')

function VolumetricUpSamplingNearest:__init(ratio)
  parent.__init(self)
  assert(math.floor(ratio) == ratio, 'ratio must be integer!')
  assert(ratio > 0, 'ratio must be non-zero positive')
  self.ratio = ratio
end

function VolumetricUpSamplingNearest:updateOutput(input)
  assert(torch.isTensor(input))
  assert(input:dim() == 5, 'Only batch mode is supported for now.')

  self.output:resize(input:size(1), input:size(2), input:size(3) * self.ratio,
                     input:size(4) * self.ratio, input:size(5) * self.ratio)
  input.tfluids.volumetricUpSamplingNearestForward(
      self.ratio, input, self.output)

  return self.output
end

function VolumetricUpSamplingNearest:updateGradInput(input, gradOutput)
  assert(torch.isTensor(input))
  assert(input:dim() == 5, 'Only batch mode is supported for now.')

  self.gradInput:resizeAs(input)
  input.tfluids.volumetricUpSamplingNearestBackward(
      self.ratio, input, gradOutput, self.gradInput)

  return self.gradInput
end

function VolumetricUpSamplingNearest:__tostring__()
  return 'VolumetricUpSamplingNearest: ratio=' .. self.ratio
end
