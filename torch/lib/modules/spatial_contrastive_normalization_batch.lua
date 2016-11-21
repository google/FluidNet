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

-- This version of SpatialConvtrastiveNormalization batch supports batches and
-- the Divisive and Subtractive modules use SpatialConvolutionMM so it's faster.
-- You will need to dofile('spatial_divisive_normalization_batch.lua') and
-- dofile('spatial_subtractive_normalization_batch.lua').

local SpatialContrastiveNormalizationBatch, parent =
  torch.class('nn.SpatialContrastiveNormalizationBatch', 'nn.Module')

function SpatialContrastiveNormalizationBatch:__init(nInputPlane, kernel,
  threshold, thresval)
  parent.__init(self)

  -- get args.
  self.nInputPlane = nInputPlane or 1
  self.kernel = kernel or torch.Tensor(9, 9):fill(1)
  self.threshold = threshold or 1e-4
  self.thresval = thresval or threshold or 1e-4
  local kdim = self.kernel:nDimension()

  assert(kdim == 2, 'No longer supporting 1D kernels!')

  -- check args.
  if (self.kernel:size(1) % 2) == 0 or (self.kernel:size(2) % 2) == 0 then
    error('<SpatialContrastiveNormalizationBatch> averaging kernel must have' ..
      ' ODD dimensions')
  end

  -- instantiate sub+div normalization.
  self.modules = {nn.Sequential()}
  self.modules[1]:add(
      nn.SpatialSubtractiveNormalizationBatch(self.nInputPlane, self.kernel))
  self.modules[1]:add(
      nn.SpatialDivisiveNormalizationBatch(self.nInputPlane, self.kernel,
                                           self.threshold, self.thresval))
end

function SpatialContrastiveNormalizationBatch:updateOutput(input)
   self.output = self.modules[1]:forward(input)
   return self.output
end

function SpatialContrastiveNormalizationBatch:updateGradInput(input, gradOutput)
   self.gradInput = self.modules[1]:backward(input, gradOutput)
   return self.gradInput
end

