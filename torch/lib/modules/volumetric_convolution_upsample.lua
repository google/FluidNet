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

-- Same as VolumetricConvolutionUpsample but for 3D convolutions.

local VolumetricConvolutionUpsample, parent =
  torch.class('nn.VolumetricConvolutionUpsample', 'nn.Module')

function VolumetricConvolutionUpsample:__init(
    nInputPlane, nOutputPlane, kT, kW, kH, dT, dW, dH, padT, padW, padH, scaleT,
    scaleW, scaleH)
  parent.__init(self)
  self.nInputPlane = nInputPlane
  self.nOutputPlane = nOutputPlane
  self.kT = kT
  self.kW = kW
  self.kH = kH
  self.dT = dT
  self.dW = dW
  self.dH = dH
  self.padT = padT
  self.padW = padW
  self.padH = padH
  self.scaleT = scaleT
  self.scaleW = scaleW
  self.scaleH = scaleH

  self.modules = {}
  self.modules[1] = nn.VolumetricConvolution(
      nInputPlane, scaleT * scaleW * scaleH * nOutputPlane, kT, kW, kH,
      dT, dW, dH, padT, padW, padH)

  self._convGradOutput = torch.Tensor()
end

function VolumetricConvolutionUpsample:updateOutput(input)
  assert(input:dim() == 5, 'Only batch mode is supported for now.')
  local convOut = self.modules[1]:updateOutput(input)
  local batchSz = convOut:size(1)
  local depth = convOut:size(3)
  local width = convOut:size(5)
  local height = convOut:size(4)
  convOut = convOut:view(batchSz, self.nOutputPlane, self.scaleT, self.scaleH,
                         self.scaleW, depth, height, width)

  self.output:resize(batchSz, self.nOutputPlane,
                     self.scaleT * depth,
                     self.scaleH * height,
                     self.scaleW * width)

  -- We're going to go from:
  -- (b, nO, sT, sH, sW, d, h, w) with indices (1, 2, 3, 4, 5, 6, 7, 8)
  -- to:
  -- (b, nO, d, sT, h, sH, w, sW) with indices (1, 2, 6, 3, 7, 4, 8, 5)

  local convOutPermuted =  convOut:permute(1, 2, 6, 3, 7, 4, 8, 5)
  assert(convOutPermuted:storage() == convOut:storage())
  local outputView = self.output:view(batchSz, self.nOutputPlane,
                                      depth, self.scaleT,
                                      height, self.scaleH,
                                      width, self.scaleW)
  assert(outputView:isSameSizeAs(convOutPermuted))
  outputView:copy(convOutPermuted)  -- copy to make contiguous.

  return self.output
end

function VolumetricConvolutionUpsample:_permuteGradOutput(gradOutput)
  assert(gradOutput:isSameSizeAs(self.output))
  local batchSz = gradOutput:size(1)
  assert(math.fmod(gradOutput:size(5), self.scaleW) == 0)
  assert(math.fmod(gradOutput:size(4), self.scaleH) == 0)
  assert(math.fmod(gradOutput:size(3), self.scaleT) == 0)
  assert(self.scaleW > 0 and self.scaleH > 0 and self.scaleT > 0)
  local width = gradOutput:size(5) / self.scaleW
  local height = gradOutput:size(4) / self.scaleH
  local depth = gradOutput:size(3) / self.scaleT
  -- Firstly permute back.
  local gradOutputView = gradOutput:view(batchSz, self.nOutputPlane,
                                         depth, self.scaleT,
                                         height, self.scaleH,
                                         width, self.scaleW)
  -- We're going to go from:
  -- (b, nO, d, sT, h, sH, w, sW) with indices (1, 2, 3, 4, 5, 6, 7, 8)
  -- to:
  -- (b, nO, sT, sH, sW, d, h, w) with indices (1, 2, 4, 6, 8, 3, 5, 7)
  local gradOutputPermuted = gradOutputView:permute(1, 2, 4, 6, 8, 3, 5, 7)
  -- Again, just make sure the permuted storage didn't change.
  assert(gradOutput:storage() == gradOutputPermuted:storage())
  self._convGradOutput:resize(batchSz, self.nOutputPlane,
                              self.scaleT, self.scaleH, self.scaleW,
                              depth, height, width)
  assert(self._convGradOutput:isSameSizeAs(gradOutputPermuted))
  self._convGradOutput:copy(gradOutputPermuted)  -- copy to make contiguous.
  local convGradOutputView = self._convGradOutput:view(
      batchSz, self.nOutputPlane * self.scaleT * self.scaleH * self.scaleW,
      depth, height, width)
  return convGradOutputView
end


function VolumetricConvolutionUpsample:updateGradInput(input, gradOutput)
  self._convGradOutput = self:_permuteGradOutput(gradOutput)
  self.gradInput = self.modules[1]:updateGradInput(input, self._convGradOutput)
  return self.gradInput
end

function VolumetricConvolutionUpsample:accGradParameters(
    input, gradOutput, scale)
  -- Assume updateGradInput has already been called.
  self.modules[1]:accGradParameters(input, self._convGradOutput, scale)
end

function VolumetricConvolutionUpsample:accUpdateGradParameters(
    input, gradOutput, lr)
  -- Assume updateGradInput has already been called.
  self.modules[1]:accUpdateGradParameters(input, self._convGradOutput, lr)
end

function VolumetricConvolutionUpsample:sharedAccUpdateGradParameters(
    input, gradOutput, lr)
  -- Assume updateGradInput has already been called.
  self.modules[1]:sharedAccUpdateGradParameters(input, self._convGradOutput, lr)
end

function VolumetricConvolutionUpsample:type(type)
  parent.type(self, type)
  self.modules[1]:type(type)
  self.gradInput = self.modules[1].gradInput
  return self
end

-- We need to return the CHILD parameters, not our references to their
-- parameters because these may become stale on type conversions.
function VolumetricConvolutionUpsample:parameters()
  return self.modules[1]:parameters()
end

function VolumetricConvolutionUpsample:clearState()
  parent.clearState(self)
  self.modules[1]:clearState()
  self._convGradOutput:set()
  return self
end

function VolumetricConvolutionUpsample:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kT, self.kW, self.kH)
   s = s .. string.format(', d: %d, %d, %d', self.dT, self.dW, self.dH)
   s = s .. ', pad: ' .. self.padW .. ', ' .. self.padH .. ', ' .. self.padT
   s = s .. ', scale: ' .. self.scaleW .. ', ' .. self.scaleH .. ', ' ..
       self.scaleT
   return s .. ')'
end
