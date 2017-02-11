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

-- This module calculates the FD approximation of the pressure gradient and
-- subtracts it from the input velocity. It is a torch replica of the
-- correctVelocity function in Manta.

local nn = require('nn')

local VelocityUpdate, parent = torch.class('tfluids.VelocityUpdate',
                                               'nn.Module')

function VelocityUpdate:__init()
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor(), torch.Tensor()}
end

function VelocityUpdate:updateOutput(input)
  assert(torch.type(input) == 'table' and #input == 3)
  local p = input[1]
  local U = input[2]
  local flags = input[3]

  self.output:resizeAs(U):copy(U)

  tfluids.velocityUpdateForward(self.output, flags, p)
  return self.output 
end

function VelocityUpdate:updateGradInput(input, gradOutput)
  local p = input[1]
  local U = input[2]
  local flags = input[3]

  self.gradInput[1]:resizeAs(p)
  self.gradInput[2]:resizeAs(U):fill(0)
  self.gradInput[3]:resizeAs(flags):fill(0)

  local gradP = self.gradInput[1]

  tfluids.velocityUpdateBackward(U, flags, p, gradOutput, gradP)
  return self.gradInput
end

function VelocityUpdate:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function VelocityUpdate:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function VelocityUpdate:type(type)
  parent.type(self, type)
  self.gradInput[1] = self.gradInput[1]:type(type)
  self.gradInput[2] = self.gradInput[2]:type(type)
  self.gradInput[3] = self.gradInput[3]:type(type)
  return self
end

function VelocityUpdate:clearState()
  -- parent.clearState(self)  -- EDIT(tompson): This will empty the table.
  self.output:set()
  self.gradInput[1]:set()
  self.gradInput[2]:set()
  self.gradInput[3]:set()
  return self
end

