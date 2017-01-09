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

-- We need to convert the Manta "flags" representation into a 0, 1 occupancy
-- grid. We also check to make sure that cells are not of another type.
-- Note: We do NOT BPROP through this cell (we just pass zero gradients).

local nn = require('nn')

local FlagsToOccupancy, parent = torch.class('tfluids.FlagsToOccupancy',
                                             'nn.Module')

function FlagsToOccupancy:__init()
  parent.__init(self)
end

function FlagsToOccupancy:updateOutput(input)
  assert(input:dim() == 5 and input:size(2) == 1)  -- Should be a 5D scalar.
  self.output:resizeAs(input)
  tfluids.flagsToOccupancy(input, self.output)
  return self.output 
end

function FlagsToOccupancy:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):fill(0)
  return self.gradInput
end

function FlagsToOccupancy:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function FlagsToOccupancy:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function FlagsToOccupancy:type(type)
  parent.type(self, type)
  return self
end

function FlagsToOccupancy:clearState()
  parent.clearState(self)
  return self
end

