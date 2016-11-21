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

-- This version of SpatialDivisiveNormalization supports batches and uses
-- the faster SpatialConvolutionMM module.

local SpatialDivisiveNormalizationBatch, parent =
  torch.class('nn.SpatialDivisiveNormalizationBatch', 'nn.Module')

-- Constants to define table location of submodules.
local MEAN_ESTIMATOR = 1
local STD_ESTIMATOR = 2
local NORMALIZER = 3
local DIVIDER = 4
local THRESHOLDER = 5

function SpatialDivisiveNormalizationBatch:__init(nInputPlane, kernel,
  threshold, thresval)
  parent.__init(self)

  -- get args.
  self.nInputPlane = nInputPlane or 1
  self.kernel = kernel or torch.Tensor(9, 9):fill(1)
  self.threshold = threshold or 1e-4
  self.thresval = thresval or threshold or 1e-4
  local kdim = self.kernel:nDimension()

   -- check args.
  if kdim ~= 2 then
    error('<SpatialDivisiveNormalizationBatch> averaging kernel must be 2D')
  end
  if (self.kernel:size(1) % 2) == 0 or (self.kernel:size(2) % 2) == 0 then
    error('<SpatialDivisiveNormalizationBatch> averaging kernel must have ' ..
      'ODD dimensions')
  end

  -- padding values.
  local padH = math.floor(self.kernel:size(1) / 2)
  local padW = math.floor(self.kernel:size(2) / 2)

  -- create convolutional mean estimator.
  self.modules = {}

  self.modules[MEAN_ESTIMATOR] = nn.Sequential()
  self.modules[MEAN_ESTIMATOR]:add(nn.SpatialZeroPadding(padW, padW, padH,
                                                         padH))
  self.modules[MEAN_ESTIMATOR]:add(cudnn.SpatialConvolution(
      self.nInputPlane, 1, self.kernel:size(2), self.kernel:size(1)))
  self.modules[MEAN_ESTIMATOR]:add(nn.Replicate(self.nInputPlane))
  self.modules[MEAN_ESTIMATOR]:add(nn.Transpose({1, 2}))

  -- create convolutional std estimator.
  self.modules[STD_ESTIMATOR] = nn.Sequential()
  self.modules[STD_ESTIMATOR]:add(nn.Square())
  self.modules[STD_ESTIMATOR]:add(nn.SpatialZeroPadding(padW, padW, padH, padH))
  self.modules[STD_ESTIMATOR]:add(cudnn.SpatialConvolution(
      self.nInputPlane, 1, self.kernel:size(2), self.kernel:size(1)))
  self.modules[STD_ESTIMATOR]:add(nn.Replicate(self.nInputPlane))
  self.modules[STD_ESTIMATOR]:add(nn.Sqrt())
  self.modules[STD_ESTIMATOR]:add(nn.Transpose({1, 2}))

  -- set kernel and bias.
  -- This is a little lazy, but create a temporary weight matrix of the
  -- original size of SpatialConvolution, and then copy it directly over
  -- to the SpatialConvolutionMM module.
  local weight = torch.Tensor(1, nInputPlane, self.kernel:size(1),
                              self.kernel:size(2))
  self.kernel:div(self.kernel:sum() * self.nInputPlane)
  for i = 1, self.nInputPlane do
    weight[1][i]:copy(self.kernel)
  end
  self.modules[MEAN_ESTIMATOR].modules[STD_ESTIMATOR].weight:copy(weight)
  self.modules[STD_ESTIMATOR].modules[3].weight:copy(weight)
  self.modules[MEAN_ESTIMATOR].modules[STD_ESTIMATOR].bias:zero()
  self.modules[STD_ESTIMATOR].modules[3].bias:zero()

  -- other operation.
  self.modules[NORMALIZER] = nn.CDivTable()
  self.modules[DIVIDER] = nn.CDivTable()
  self.modules[THRESHOLDER] = nn.Threshold(self.threshold, self.thresval)

  -- coefficient array, to adjust side effects.
  self.coef = torch.Tensor(1, 1, 1, 1)
end

function SpatialDivisiveNormalizationBatch:updateOutput(input)
  -- compute side coefficients.
  if (input:size(4) ~= self.coef:size(4)) or
     (input:size(3) ~= self.coef:size(3)) or
     (input:size(1) ~= self.coef:size(1)) then
    local ones = input.new():resizeAs(input):fill(1)
    self.coef = self.modules[MEAN_ESTIMATOR]:updateOutput(ones)
    self.coef = self.coef:clone()
  end

  -- normalize std dev.
  self.localStds = self.modules[STD_ESTIMATOR]:updateOutput(input)
  self.adjustedStds =
      self.modules[DIVIDER]:updateOutput{self.localStds, self.coef}
  self.thresholdedStds =
      self.modules[THRESHOLDER]:updateOutput(self.adjustedStds)
  self.output =
      self.modules[NORMALIZER]:updateOutput{input, self.thresholdedStds}

  return self.output
end

function SpatialDivisiveNormalizationBatch:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  -- backprop through all modules.
  local gradNorm =
      self.modules[NORMALIZER]:updateGradInput({input, self.thresholdedStds},
                                               gradOutput)
  local gradAdj =
      self.modules[THRESHOLDER]:updateGradInput(self.adjustedStds, gradNorm[2])
  local gradDiv =
      self.modules[DIVIDER]:updateGradInput({self.localStds, self.coef},
                                            gradAdj)
  self.gradInput:add(
      self.modules[STD_ESTIMATOR]:updateGradInput(input, gradDiv[1]))
  self.gradInput:add(gradNorm[1])

  return self.gradInput
end

