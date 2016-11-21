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

-- This module takes in a 4D tensor of Batch x Feats x Height x Width
-- and calculates the central difference difference of dFeats / dWidth and
-- dFeats / dHeight (i.e. the partial derivative of each feature slice w.r.t.
-- the spatial dimensions.). The border pixels are calculated using single sided
-- finite difference.
--
-- The output is size: Batch x Feats x 2 x Height x Width
-- Where: - index 1 of dim 3 is the dF/dx (width dimension)
--        - index 2 of dim 3 is the dF/dy (height dimension)

local SpatialFiniteElements, parent =
  torch.class('nn.SpatialFiniteElements', 'nn.Module')

-- @param stepSizeX/Y - OPTIONAL - as with gradient function in Matlab, the user
-- can specify the grid step size in the X and Y dimensions. Default is 1.
function SpatialFiniteElements:__init(stepSizeX, stepSizeY)
  parent.__init(self)
  self.stepSizeX = stepSizeX or 1
  self.stepSizeY = stepSizeY or 1

  assert(self.stepSizeX > 0 and self.stepSizeY > 0)

  local horiz = nn.Sequential()
  horiz:add(nn.SpatialReplicationPadding(1, 1, 0, 0))  -- l, r, t, b
  local conv = nn.SpatialConvolution(1, 1, 3, 1)
  horiz:add(conv)  -- 3 x 1 convolution
  conv.bias:fill(0)
  conv.weight[{1, 1, 1, 1}] = -1 / (2 * self.stepSizeX)
  conv.weight[{1, 1, 1, 2}] = 0
  conv.weight[{1, 1, 1, 3}] = 1 / (2 * self.stepSizeX)
  self.horiz = horiz

  local vert = nn.Sequential()
  vert:add(nn.SpatialReplicationPadding(0, 0, 1, 1))  -- l, r, t, b 
  conv = nn.SpatialConvolution(1, 1, 1, 3)
  vert:add(conv)
  conv.bias:fill(0)
  conv.weight[{1, 1, 1, 1}] = -1 / (2 * self.stepSizeY)
  conv.weight[{1, 1, 2, 1}] = 0
  conv.weight[{1, 1, 3, 1}] = 1 / (2 * self.stepSizeY)
  self.vert = vert

  self._gradOutputBuffer = torch.Tensor()
  self._gradOutHoriz1f = torch.Tensor()
  self._gradOutVert1f = torch.Tensor()
end

local function multBorderPixels(tensor, scale)
  assert(tensor:dim() == 5)
  assert(tensor:size(3) == 2)
  tensor[{{}, {}, 1, {}, 1}]:mul(scale)
  tensor[{{}, {}, 1, {}, -1}]:mul(scale)
  tensor[{{}, {}, 2, 1, {}}]:mul(scale)
  tensor[{{}, {}, 2, -1, {}}]:mul(scale)
end

function SpatialFiniteElements:updateOutput(input)
  assert(input:dim() == 4)
  local nbatch = input:size(1)
  local f = input:size(2)
  local h = input:size(3)
  local w = input:size(4)
  self.output:resize(nbatch, f, 2, h, w)
  
  local input1f = input:view(nbatch * f, 1, h, w)
  local outHoriz = self.horiz:updateOutput(input1f):view(nbatch, f, h, w)
  local outVert = self.vert:updateOutput(input1f):view(nbatch, f, h, w)
  
  self.output[{{}, {}, 1, {}, {}}]:copy(outHoriz)
  self.output[{{}, {}, 2, {}, {}}]:copy(outVert) 

  -- We're almost correct, however the derivative on the border pixels is
  -- of by 2x. (this is because we did a clamped padding to add the extra
  -- pixels, but then the central difference term has a 1/2).
  multBorderPixels(self.output, 2)

  return self.output 
end

function SpatialFiniteElements:updateGradInput(input, gradOutput)
  assert(input:dim() == 4)
  assert(gradOutput:dim() == 5)
  assert(gradOutput:size(1) == input:size(1) and
         gradOutput:size(2) == input:size(2) and
         gradOutput:size(3) == 2 and
         gradOutput:size(4) == input:size(3) and
         gradOutput:size(5) == input:size(4))
  local nbatch = input:size(1)
  local f = input:size(2)
  local h = input:size(3)
  local w = input:size(4)
  self.gradInput:resizeAs(input)
  
  -- We had to multiply the border pixels by a factor of 2. So multiply the
  -- gradOutput border pixels by 2 to match FPROP.
  self._gradOutputBuffer:resizeAs(gradOutput)
  self._gradOutputBuffer:copy(gradOutput)
  multBorderPixels(self._gradOutputBuffer, 2)
  gradOutput = self._gradOutputBuffer

  local input1f = input:view(nbatch * f, 1, h, w)
  local gradOutHoriz = gradOutput[{{}, {}, 1, {}, {}}]
  -- The copy here is to force gradOutputHoriz1f to be contiguous.
  self._gradOutHoriz1f:resize(nbatch * f, 1, h, w)
  self._gradOutHoriz1f:copy(gradOutHoriz)
  local gradOutVert = gradOutput[{{}, {}, 2, {}, {}}]
  self._gradOutVert1f:resize(nbatch * f, 1, h, w)
  self._gradOutVert1f:copy(gradOutVert)

  local gradInputHoriz1f =
      self.horiz:updateGradInput(input1f, self._gradOutHoriz1f)
  local gradInputHoriz = gradInputHoriz1f:view(nbatch, f, h, w)
  local gradInputVert1f =
      self.vert:updateGradInput(input1f, self._gradOutVert1f)
  local gradInputVert = gradInputVert1f:view(nbatch, f, h, w)

  -- Finally, add the vertical and horizontal gradients.
  self.gradInput:copy(gradInputHoriz):add(gradInputVert)
 
  return self.gradInput
end

function SpatialFiniteElements:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function SpatialFiniteElements:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function SpatialFiniteElements:type(type)
  parent.type(self, type)
  self.horiz:type(type)
  self.vert:type(type)
  return self
end

function SpatialFiniteElements:clearState()
  parent.clearState(self)
  self._gradOutputBuffer:set()
  self._gradOutHoriz1f:set()
  self._gradOutVert1f:set()
  self.horiz:clearState()
  self.vert:clearState()
  return self
end

