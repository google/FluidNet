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

-- This module calculates the FD approximation of the divergence of a velocity
-- field (MAC grid).

local nn = require('nn')

local VelocityDivergence, parent = torch.class('tfluids.VelocityDivergence',
                                               'nn.Module')

function VelocityDivergence:__init()
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function VelocityDivergence:updateOutput(input)
  assert(torch.type(input) == 'table' and #input == 2)
  local U = input[1]
  local flags = input[2]

  self.output:resizeAs(flags)

  tfluids.velocityDivergenceForward(U, flags, self.output)
  return self.output 
end

function VelocityDivergence:updateGradInput(input, gradOutput)
  local U = input[1]
  local flags = input[2]

  self.gradInput[1]:resizeAs(U)
  self.gradInput[2]:resizeAs(flags):fill(0)  -- Fill with 0.

  local gradU = self.gradInput[1]

  tfluids.velocityDivergenceBackward(U, flags, gradOutput, gradU)
  return self.gradInput
end

function VelocityDivergence:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function VelocityDivergence:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function VelocityDivergence:type(type)
  parent.type(self, type)
  self.gradInput[1] = self.gradInput[1]:type(type)
  self.gradInput[2] = self.gradInput[2]:type(type)
  return self
end

function VelocityDivergence:clearState()
  -- parent.clearState(self)  -- EDIT(tompson): This will empty the table.
  self.output:set()
  self.gradInput[1]:set()
  self.gradInput[2]:set()
  return self
end

