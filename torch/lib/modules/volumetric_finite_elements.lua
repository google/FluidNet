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

-- This module takes in a 5D tensor of Batch x Feats x Depth x Height x Width
-- and calculates the central difference difference of dFeats / dWidth,
-- dFeats / dHeight and dFeats / dDepth (i.e. the partial derivative of each
-- feature slice w.r.t. the volume dimensions.). The border pixels are
-- calculated using single sided finite difference.
--
-- The output is size: Batch x Feats x 3 x Depth x Height x Width
-- Where: - index 1 of dim 3 is the dF/dx (width dimension)
--        - index 2 of dim 3 is the dF/dy (height dimension)
--        - index 3 of dim 3 is the dF/dz (depth dimension)

local VolumetricFiniteElements, parent =
  torch.class('nn.VolumetricFiniteElements', 'nn.Module')

-- @param stepSizeX/Y/Z - OPTIONAL - as with gradient function in Matlab, the
-- user can specify the grid step size in the X and Y dimensions. Default is 1.
function VolumetricFiniteElements:__init(stepSizeX, stepSizeY, stepSizeZ)
  parent.__init(self)
  self.stepSizeX = stepSizeX or 1
  self.stepSizeY = stepSizeY or 1
  self.stepSizeZ = stepSizeZ or 1

  assert(self.stepSizeX > 0 and self.stepSizeY > 0 and self.stepSizeZ > 0)
  
  local horiz = nn.Sequential()
  local pad = {1, 1, 0, 0, 0, 0}  -- left, right, top, bottom, front, back
  horiz:add(nn.VolumetricReplicationPadding(unpack(pad)))
  local conv = nn.VolumetricConvolution(1, 1, 1, 3, 1)  -- ip, op, kt, kw, kh
  horiz:add(conv)  -- 1 x 3 x 1 convolution (d x w x h)
  conv.bias:fill(0)
  conv.weight[{1, 1, 1, 1, 1}] = -1 / (2 * self.stepSizeX)
  conv.weight[{1, 1, 1, 1, 2}] = 0
  conv.weight[{1, 1, 1, 1, 3}] = 1 / (2 * self.stepSizeX)
  self.horiz = horiz

  local vert = nn.Sequential()
  pad = {0, 0, 1, 1, 0, 0}  -- left, right, top, bottom, front, back
  vert:add(nn.VolumetricReplicationPadding(unpack(pad)))
  conv = nn.VolumetricConvolution(1, 1, 1, 1, 3)
  vert:add(conv)  -- 1 x 1 x 3 convolution (d x w x h)
  conv.bias:fill(0)
  conv.weight[{1, 1, 1, 1, 1}] = -1 / (2 * self.stepSizeY)
  conv.weight[{1, 1, 1, 2, 1}] = 0
  conv.weight[{1, 1, 1, 3, 1}] = 1 / (2 * self.stepSizeY)
  self.vert = vert

  local depth = nn.Sequential()
  pad = {0, 0, 0, 0, 1, 1}  -- left, right, top, bottom, front, back
  depth:add(nn.VolumetricReplicationPadding(unpack(pad)))
  conv = nn.VolumetricConvolution(1, 1, 3, 1, 1)
  depth:add(conv)  -- 3 x 1 x 1 convolution (d x w x h)
  conv.bias:fill(0)
  conv.weight[{1, 1, 1, 1, 1}] = -1 / (2 * self.stepSizeZ)
  conv.weight[{1, 1, 2, 1, 1}] = 0
  conv.weight[{1, 1, 3, 1, 1}] = 1 / (2 * self.stepSizeZ)
  self.depth = depth

  self._gradOutputBuffer = torch.Tensor()
  self._gradOutHoriz1f = torch.Tensor()
  self._gradOutVert1f = torch.Tensor()
  self._gradOutDepth1f = torch.Tensor()
end

local function multBorderPixels(tensor, scale)
  assert(tensor:dim() == 6)
  assert(tensor:size(3) == 3)
  tensor[{{}, {}, 1, {}, {}, 1}]:mul(scale)
  tensor[{{}, {}, 1, {}, {}, -1}]:mul(scale)
  tensor[{{}, {}, 2, {}, 1, {}}]:mul(scale)
  tensor[{{}, {}, 2, {}, -1, {}}]:mul(scale)
  tensor[{{}, {}, 3, 1, {}, {}}]:mul(scale)
  tensor[{{}, {}, 3, -1, {}, {}}]:mul(scale)
end

function VolumetricFiniteElements:updateOutput(input)
  assert(input:dim() == 5)
  local nbatch = input:size(1)
  local f = input:size(2)
  local d = input:size(3)
  local h = input:size(4)
  local w = input:size(5)
  self.output:resize(nbatch, f, 3, d, h, w)
  
  local input1f = input:view(nbatch * f, 1, d, h, w)
  local outHoriz = self.horiz:updateOutput(input1f):view(nbatch, f, d, h, w)
  local outVert = self.vert:updateOutput(input1f):view(nbatch, f, d, h, w)
  local outDepth = self.depth:updateOutput(input1f):view(nbatch, f, d, h, w)

  self.output[{{}, {}, 1, {}, {}, {}}]:copy(outHoriz)
  self.output[{{}, {}, 2, {}, {}, {}}]:copy(outVert)
  self.output[{{}, {}, 3, {}, {}, {}}]:copy(outDepth)

  -- We're almost correct, however the derivative on the border pixels is
  -- of by 2x. (this is because we did a clamped padding to add the extra
  -- pixels, but then the central difference term has a 1/2).
  multBorderPixels(self.output, 2)

  return self.output 
end

function VolumetricFiniteElements:updateGradInput(input, gradOutput)
  assert(input:dim() == 5)
  assert(gradOutput:dim() == 6)
  assert(gradOutput:size(1) == input:size(1) and
         gradOutput:size(2) == input:size(2) and
         gradOutput:size(3) == 3 and
         gradOutput:size(4) == input:size(3) and
         gradOutput:size(5) == input:size(4) and
         gradOutput:size(6) == input:size(5))
  local nbatch = input:size(1)
  local f = input:size(2)
  local d = input:size(3)
  local h = input:size(4)
  local w = input:size(5)
  self.gradInput:resizeAs(input)
  
  -- We had to multiply the border pixels by a factor of 2. So multiply the
  -- gradOutput border pixels by 2 to match FPROP.
  self._gradOutputBuffer:resizeAs(gradOutput)
  self._gradOutputBuffer:copy(gradOutput)
  multBorderPixels(self._gradOutputBuffer, 2)
  gradOutput = self._gradOutputBuffer

  local input1f = input:view(nbatch * f, 1, d, h, w)
  local gradOutHoriz = gradOutput[{{}, {}, 1, {}, {}, {}}]
  -- The copy here is to force gradOutputHoriz1f to be contiguous.
  self._gradOutHoriz1f:resize(nbatch * f, 1, d, h, w)
  self._gradOutHoriz1f:copy(gradOutHoriz)
  local gradOutVert = gradOutput[{{}, {}, 2, {}, {}, {}}]
  self._gradOutVert1f:resize(nbatch * f, 1, d, h, w)
  self._gradOutVert1f:copy(gradOutVert)
  local gradOutDepth = gradOutput[{{}, {}, 3, {}, {}, {}}]
  self._gradOutDepth1f:resize(nbatch * f, 1, d, h, w)
  self._gradOutDepth1f:copy(gradOutDepth)

  local gradInputHoriz1f =
      self.horiz:updateGradInput(input1f, self._gradOutHoriz1f)
  local gradInputHoriz = gradInputHoriz1f:view(nbatch, f, d, h, w)
  local gradInputVert1f =
      self.vert:updateGradInput(input1f, self._gradOutVert1f)
  local gradInputVert = gradInputVert1f:view(nbatch, f, d, h, w)
  local gradInputDepth1f =
      self.depth:updateGradInput(input1f, self._gradOutDepth1f)
  local gradInputDepth = gradInputDepth1f:view(nbatch, f, d, h, w)

  -- Finally, add the vertical, horizontal and depth gradients.
  self.gradInput:copy(gradInputHoriz):add(gradInputVert):add(gradInputDepth)
 
  return self.gradInput
end

function VolumetricFiniteElements:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function VolumetricFiniteElements:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function VolumetricFiniteElements:type(type)
  parent.type(self, type)
  self.horiz:type(type)
  self.vert:type(type)
  self.depth:type(type)
  return self
end

function VolumetricFiniteElements:clearState()
  parent.clearState(self)
  self._gradOutputBuffer:set()
  self._gradOutHoriz1f:set()
  self._gradOutVert1f:set()
  self._gradOutDepth1f:set()
  self.horiz:clearState()
  self.vert:clearState()
  self.depth:clearState()
  return self
end

