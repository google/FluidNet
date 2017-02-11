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

-- Simply a nn wrapper around tfluids.setWallBcs.
--
-- The inputs must be batched.

local nn = require('nn')

local SetWallBcs, parent = torch.class('tfluids.SetWallBcs', 'nn.Module')

function SetWallBcs:__init()
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self.mask = torch.Tensor()
end

function SetWallBcs:updateOutput(input)
  assert(torch.type(input) == 'table' and #input == 2)
  local U = input[1]
  local flags = input[2]

  self.output:resizeAs(U)
  self.output:copy(U)

  -- Create a 0 mask (i.e. solid velocities will be set to zero). Doing it this
  -- way means that we can reuse this mask during BPROP.
  self.mask:resizeAs(U):fill(1)
  tfluids.setWallBcsForward(self.mask, flags)

  -- TODO(tompson): if we ever switch to using non-zero obstacle velocities
  -- this "mask strategy" will not work.

  self.output:cmul(self.mask)

  return self.output 
end

function SetWallBcs:updateGradInput(input, gradOutput)
  -- Assume updateOutput has been called.
  local U = input[1]
  local flags = input[2]

  self.gradInput[1]:resizeAs(gradOutput):fill(0)
  self.gradInput[2]:resizeAs(flags):fill(0)  -- Fill with 0.

  -- A trick: the forward call just zeros out velocities based on the flag
  -- grid. Therefore we can simply copy the gradOutput and treat it as an
  -- input velocity and call the forward function.

  self.gradInput[1]:addcmul(1, self.mask, gradOutput)

  return self.gradInput
end

function SetWallBcs:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function SetWallBcs:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function SetWallBcs:type(type)
  parent.type(self, type)
  for i = 1, #self.gradInput do
    self.gradInput[i] = self.gradInput[i]:type(type)
  end
  return self
end

function SetWallBcs:clearState()
  parent.clearState(self)
  -- The parent clearState call will (for some reason) completely clear the
  -- gradInput table, so we have to re-allocate the tensors.
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  for i = 1, #self.gradInput do
    self.gradInput[i] = self.gradInput[i]:type(torch.type(self.output))
  end
  self.mask:set()
end

