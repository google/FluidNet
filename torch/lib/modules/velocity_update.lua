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

-- This module simply calls the tfluids.calcVelocityUpdate on forward and
-- tfluids.calcVelocityUpdateBackward on backwards.
--
-- The module takes in a table of {p, geom}, each of size
-- (batch, 1, depth, height, width) and outputs deltaU of size
-- (batch, 2/3, depth, height, width).

local tfluids = require('tfluids')

local VelocityUpdate, parent =
  torch.class('nn.VelocityUpdate', 'nn.Module')

function VelocityUpdate:__init(matchManta)
  parent.__init(self)
  if matchManta == nil then
    self.matchManta = false
  else
    self.matchManta = matchManta
  end
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function VelocityUpdate:updateOutput(input)
  assert(torch.type(input) == 'table' and #input == 2)
  local p = input[1]
  local geom = input[2]

  assert(p:dim() == 5 or p:dim() == 4)  -- OLD style: 4D, NEW style: 5D.
  assert(geom:isSameSizeAs(p))

  if p:dim() == 5 then
    p = p[{{}, 1}]  -- Remove the singleton dimension
    geom = geom[{{}, 1}]  -- Remove the singleton dimension.
  end

  local twoDim = p:size(2) == 1
  if twoDim then
    self.output:resize(p:size(1), 2, p:size(2), p:size(3), p:size(4))
  else
    self.output:resize(p:size(1), 3, p:size(2), p:size(3), p:size(4))
  end

  tfluids.calcVelocityUpdate(self.output, p, geom, self.matchManta)
  return self.output 
end

function VelocityUpdate:updateGradInput(input, gradOutput)
  -- Assume updateOutput has already been called.
  assert(self.output:isSameSizeAs(gradOutput))
  local p = input[1]
  local geom = input[2]

  self.gradInput[1]:resizeAs(p)
  self.gradInput[2]:resizeAs(geom):fill(0)  -- Fill with 0.

  local gradP = self.gradInput[1]

  if p:dim() == 5 then
    p = p[{{}, 1}]  -- Remove the singleton dimension
    geom = geom[{{}, 1}]  -- Remove the singleton dimension.
    gradP = gradP[{{}, 1}]  -- Remove the singleton dimension.
  end

  tfluids.calcVelocityUpdateBackward(gradP, p, geom, gradOutput,
                                     self.matchManta)
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
  for i = 1, #self.gradInput do
    self.gradInput[i] = self.gradInput[i]:type(type)
  end
  return self
end

function VelocityUpdate:clearState()
  parent.clearState(self)
  -- The parent clearState call will (for some reason) completely clear the
  -- gradInput table, so we have to re-allocate the tensors.
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  for i = 1, #self.gradInput do
    self.gradInput[i] = self.gradInput[i]:type(torch.type(self.output))
  end
end

