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

-- This module is a convolution layer that ALSO performs spatial upsampling by
-- increasing the size of the feature domain representation, then interleaving
-- feature pixels in the spatial domain. You can view this in a few ways: 1. It
-- is "like" a learned "template model" or a linear basis set for upsampling, or
-- 2. It is an interpolation stage where the interpolant input context is the
-- size of the convolution window and the interpolation kernel is learned per
-- input & output feature.

local SpatialConvolutionUpsample, parent =
  torch.class('nn.SpatialConvolutionUpsample', 'nn.Module')

function SpatialConvolutionUpsample:__init(nInputPlane, nOutputPlane, kW, kH,
                                           dW, dH, padW, padH, scaleW, scaleH)
  parent.__init(self)
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.kW = kW
  self.kH = kH
  self.dW = dW
  self.dH = dH
  self.padW = padW
  self.padH = padH
  self.scaleW = scaleW
  self.scaleH = scaleH

  self.modules = {}
  self.modules[1] = nn.SpatialConvolution(
      nInputPlane, scaleW * scaleH * nOutputPlane, kW, kH, dW, dH, padW, padH)

  self._convGradOutput = torch.Tensor()
end

function SpatialConvolutionUpsample:updateOutput(input)
  assert(input:dim() == 4, 'Only batch mode is supported for now.')
  local convOut = self.modules[1]:updateOutput(input)
  local batchSz = convOut:size(1)
  local width = convOut:size(4)
  local height = convOut:size(3)
  convOut = convOut:view(batchSz, self.nOutputPlane, self.scaleH,
                         self.scaleW, height, width)

  -- Now we need to permute the dimensions of the above view.
  -- There's "some chance" we might be able to do this without an alloc and
  -- copy, but this would require hacking into CudaTensor. Since later stages
  -- will likely require a contiguous output, we should probably perform a
  -- copy anyway.
  -- Note: if we knew the input image size during __init, we could do this
  -- with nn.View and nn.Transpose layers, but since we want to be adaptable
  -- to varying input image sizes, we'll do it with pure lua commands.
  self.output:resize(batchSz, self.nOutputPlane,
                     self.scaleH * height, self.scaleW * width)

  -- We're going from (b, nO, sH, sW, h, w) with indices (1, 2, 3, 4, 5, 6).
  -- to (b, nO, h, sH, w, sW) with indices (1, 2, 5, 3, 6, 4).

  -- Another note: torch.permute does not accept pre-allocated memory for the
  -- return tensor and is actually just a sequence of 'transpose' calls anyway.
  -- So we need to then do a copy to make the tensor contiguous.
  local convOutPermuted =  convOut:permute(1, 2, 5, 3, 6, 4)
  -- Make sure this is just a view change by checking that the underlining
  -- storage is the same.
  assert(convOutPermuted:storage() == convOut:storage())
  local outputView = self.output:view(batchSz, self.nOutputPlane,
                                      height, self.scaleH, width, self.scaleW)
  assert(outputView:isSameSizeAs(convOutPermuted))
  outputView:copy(convOutPermuted)  -- copy to make contiguous.

  return self.output
end

function SpatialConvolutionUpsample:_permuteGradOutput(gradOutput)
  assert(gradOutput:isSameSizeAs(self.output))
  local batchSz = gradOutput:size(1)
  assert(math.fmod(gradOutput:size(4), self.scaleW) == 0)
  assert(math.fmod(gradOutput:size(3), self.scaleH) == 0)
  assert(self.scaleW > 0 and self.scaleH > 0)
  local width = gradOutput:size(4) / self.scaleW
  local height = gradOutput:size(3) / self.scaleH
  -- Firstly permute back.
  local gradOutputView = gradOutput:view(batchSz, self.nOutputPlane, height,
                                         self.scaleH, width, self.scaleW)
  -- We're going from (b, nO, h, sH, w, sW) with indices (1, 2, 3, 4, 5, 6).
  -- to (b, nO, sH, sW, h, w) with indices (1, 2, 4, 6, 3, 5).
  local gradOutputPermuted = gradOutputView:permute(1, 2, 4, 6, 3, 5)
  -- Again, just make sure the permuted storage didn't change.
  assert(gradOutput:storage() == gradOutputPermuted:storage())
  self._convGradOutput:resize(batchSz, self.nOutputPlane, self.scaleH,
                             self.scaleW, height, width)
  assert(self._convGradOutput:isSameSizeAs(gradOutputPermuted))
  self._convGradOutput:copy(gradOutputPermuted)  -- copy to make contiguous.
  local convGradOutputView = self._convGradOutput:view(
      batchSz, self.nOutputPlane * self.scaleH * self.scaleW, height, width)
  return convGradOutputView
end


function SpatialConvolutionUpsample:updateGradInput(input, gradOutput)
  self._convGradOutput = self:_permuteGradOutput(gradOutput)
  self.gradInput = self.modules[1]:updateGradInput(input, self._convGradOutput)
  return self.gradInput
end

function SpatialConvolutionUpsample:accGradParameters(input, gradOutput, scale)
  -- Assume updateGradInput has already been called.
  self.modules[1]:accGradParameters(input, self._convGradOutput, scale)
end

function SpatialConvolutionUpsample:accUpdateGradParameters(input, gradOutput,
                                                            lr)
  -- Assume updateGradInput has already been called.
  self.modules[1]:accUpdateGradParameters(input, self._convGradOutput, lr)
end

function SpatialConvolutionUpsample:sharedAccUpdateGradParameters(input,
                                                                  gradOutput,
                                                                  lr)
  -- Assume updateGradInput has already been called.
  self.modules[1]:sharedAccUpdateGradParameters(input, self._convGradOutput, lr)
end

function SpatialConvolutionUpsample:type(type)
  parent.type(self, type)
  self.modules[1]:type(type)
  self.gradInput = self.modules[1].gradInput
  return self
end

-- We need to return the CHILD parameters, not our references to their
-- parameters because these may become stale on type conversions.
function SpatialConvolutionUpsample:parameters()
  return self.modules[1]:parameters()
end

function SpatialConvolutionUpsample:clearState()
  parent.clearState(self)
  self.modules[1]:clearState()
  self._convGradOutput:set()
  return self
end

function SpatialConvolutionUpsample:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   s = s .. string.format(', d: %d, %d', self.dW, self.dH)
   s = s .. ', pad: ' .. self.padW .. ',' .. self.padH
   s = s .. ', scale: ' .. self.scaleW .. ', ' .. self.scaleH
   return s .. ')'
end
