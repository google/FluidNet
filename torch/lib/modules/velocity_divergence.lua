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

-- This module calculates the FEM approximation of the divergence of a velocity
-- field. If there is no geom voxels than it is central difference FEM
-- internally and single sided diff on the border. If there is geometry, then
-- it uses single sided diff to avoid sampling inside geom cells.
--
-- The module takes in a table of {U, geom} and outputs UDiv of size
-- (batch, depth, height, width).

local tfluids = require('tfluids')

local VelocityDivergence, parent =
  torch.class('nn.VelocityDivergence', 'nn.Module')

function VelocityDivergence:__init(matchManta)
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function VelocityDivergence:updateOutput(input)
  assert(torch.type(input) == 'table' and #input == 2)
  local U = input[1]
  local geom = input[2]

  assert(U:dim() == 5)  -- We include a batch dimension.
  assert(geom:dim() == 4)

  self.output:resizeAs(geom)

  tfluids.calcVelocityDivergence(U, geom, self.output)
  return self.output 
end

function VelocityDivergence:updateGradInput(input, gradOutput)
  -- Assume updateOutput has already been called.
  assert(self.output:isSameSizeAs(gradOutput))
  local U = input[1]
  local geom = input[2]

  self.gradInput[1]:resizeAs(U)
  self.gradInput[2]:resizeAs(geom):fill(0)  -- Fill with 0.

  local gradU = self.gradInput[1]

  tfluids.calcVelocityDivergenceBackward(gradU, U, geom, gradOutput)
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

