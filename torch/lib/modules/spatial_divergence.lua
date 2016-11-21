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

-- This module takes in a 4D tensor of Batch x 2 x Height x Width
-- and calculates the central difference divergence:
-- dot(grad, Tensor) = dF1 / dWidth + dF2 / dHeight.
-- The border pixels are calculated using single sided finite difference.
--
-- The output is size: Batch x 1 x Height x Width

local SpatialDivergence, parent =
  torch.class('nn.SpatialDivergence', 'nn.Module')

-- @param stepSizeX/Y - OPTIONAL - as with gradient function in Matlab, the user
-- can specify the grid step size in the X and Y dimensions. Default is 1.
function SpatialDivergence:__init(stepSizeX, stepSizeY)
  parent.__init(self)
  self.stepSizeX = stepSizeX or 1
  self.stepSizeY = stepSizeY or 1

  assert(self.stepSizeX > 0 and self.stepSizeY > 0)

  local horiz = nn.Sequential()
  horiz:add(nn.Narrow(2, 1, 1))  -- Pick off the x dimension on input.
  horiz:add(nn.SpatialReplicationPadding(1, 1, 0, 0))  -- l, r, t, b
  local conv = nn.SpatialConvolution(1, 1, 3, 1)
  horiz:add(conv)  -- 3 x 1 convolution
  conv.bias:fill(0)
  conv.weight[{1, 1, 1, 1}] = -1 / (2 * self.stepSizeX)
  conv.weight[{1, 1, 1, 2}] = 0
  conv.weight[{1, 1, 1, 3}] = 1 / (2 * self.stepSizeX)
  self.horiz = horiz

  local vert = nn.Sequential()
  vert:add(nn.Narrow(2, 2, 1))  -- Pick off the y dimension on input.
  vert:add(nn.SpatialReplicationPadding(0, 0, 1, 1))  -- l, r, t, b 
  conv = nn.SpatialConvolution(1, 1, 1, 3)
  vert:add(conv)
  conv.bias:fill(0)
  conv.weight[{1, 1, 1, 1}] = -1 / (2 * self.stepSizeY)
  conv.weight[{1, 1, 2, 1}] = 0
  conv.weight[{1, 1, 3, 1}] = 1 / (2 * self.stepSizeY)
  self.vert = vert

  self._gradOutputVert = torch.Tensor()
  self._gradOutputHoriz = torch.Tensor()
end

function SpatialDivergence:updateOutput(input)
  assert(input:dim() == 4)
  local nbatch = input:size(1)
  assert(input:size(2) == 2)
  local h = input:size(3)
  local w = input:size(4)
  self.output:resize(nbatch, 1, h, w)
  
  local outHoriz = self.horiz:updateOutput(input)
  local outVert = self.vert:updateOutput(input)
  assert(outHoriz:dim() == 4 and outVert:dim() == 4)
  assert(outHoriz:size(2) == 1 and outVert:size(2) == 1)

  -- We're almost correct, however the derivative on the border pixels is
  -- of by 2x. (this is because we did a clamped padding to add the extra
  -- pixels, but then the central difference term has a 1/2).
  outHoriz[{{}, {}, {}, 1}]:mul(2)
  outHoriz[{{}, {}, {}, -1}]:mul(2)
  outVert[{{}, {}, 1, {}}]:mul(2)
  outVert[{{}, {}, -1, {}}]:mul(2)

  self.output:copy(outHoriz):add(outVert)

  return self.output 
end

function SpatialDivergence:updateGradInput(input, gradOutput)
  assert(input:dim() == 4)
  assert(gradOutput:dim() == 4)
  assert(gradOutput:size(1) == input:size(1) and
         gradOutput:size(2) == 1 and
         gradOutput:size(3) == input:size(3) and
         gradOutput:size(4) == input:size(4))
  self.gradInput:resizeAs(input)
  
  -- We had to multiply the border pixels by a factor of 2. So multiply the
  -- gradOutput border pixels by 2 to match FPROP.
  self._gradOutputVert:resizeAs(gradOutput)
  self._gradOutputVert:copy(gradOutput)
  self._gradOutputHoriz:resizeAs(gradOutput)
  self._gradOutputHoriz:copy(gradOutput)
  self._gradOutputHoriz[{{}, {}, {}, 1}]:mul(2)
  self._gradOutputHoriz[{{}, {}, {}, -1}]:mul(2)
  self._gradOutputVert[{{}, {}, 1, {}}]:mul(2)
  self._gradOutputVert[{{}, {}, -1, {}}]:mul(2)

  local gradInputHoriz =
      self.horiz:updateGradInput(input, self._gradOutputHoriz)
  local gradInputVert =
      self.vert:updateGradInput(input, self._gradOutputVert)

  self.gradInput:copy(gradInputHoriz)
  self.gradInput:add(gradInputVert)
 
  return self.gradInput
end

function SpatialDivergence:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function SpatialDivergence:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function SpatialDivergence:type(type)
  parent.type(self, type)
  self.horiz:type(type)
  self.vert:type(type)
  return self
end

function SpatialDivergence:clearState()
  parent.clearState(self)
  self.horiz:clearState()
  self.vert:clearState()
  self.outputVert:clear()
  self.outputHoriz:clear()
end
