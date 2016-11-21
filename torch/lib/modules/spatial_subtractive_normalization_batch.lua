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

-- This version of SpatialSubtractiveNormalization supports batches and uses
-- SpatialConvolutionMM (so it's slightly faster).
local SpatialSubtractiveNormalizationBatch, parent =
  torch.class('nn.SpatialSubtractiveNormalizationBatch', 'nn.Module')

-- Constants to define table location of submodules.
local MEAN_ESTIMATOR = 1
local SUBTRACTOR = 2
local DIVIDER = 3

function SpatialSubtractiveNormalizationBatch:__init(nInputPlane, kernel)
  parent.__init(self)

  -- get args.
  self.nInputPlane = nInputPlane or 1
  self.kernel = kernel or torch.Tensor(9, 9):fill(1)
  local kdim = self.kernel:nDimension()

  -- check args.
  if kdim ~= 2 then
    error('<SpatialSubtractiveNormalizationBatch> averaging kernel must be 2D')
  end
  if (self.kernel:size(1) % 2) == 0 or (self.kernel:size(2) % 2) == 0 then
    error('<SpatialSubtractiveNormalizationBatch> averaging kernel must have' ..
      ' ODD dimensions')
  end

  -- normalize kernel.
  self.kernel:div(self.kernel:sum() * self.nInputPlane)

  -- padding values.
  local padH = math.floor(self.kernel:size(1) / 2)
  local padW = math.floor(self.kernel:size(2) / 2)

  self.modules = {}

  -- create convolutional mean extractor.
  self.modules[MEAN_ESTIMATOR] = nn.Sequential()
  self.modules[MEAN_ESTIMATOR]:add(nn.SpatialZeroPadding(padW, padW, padH,
                                                         padH))
  self.modules[MEAN_ESTIMATOR]:add(cudnn.SpatialConvolution(
      self.nInputPlane, 1, self.kernel:size(2), self.kernel:size(1)))
  self.modules[MEAN_ESTIMATOR]:add(nn.Replicate(self.nInputPlane))
  self.modules[MEAN_ESTIMATOR]:add(nn.Transpose({1, 2}))

  -- set kernel and bias.
  -- This is a little lazy, but create a temporary weight matrix of the
  -- original size of SpatialConvolution, and then copy it directly over
  -- to the SpatialConvolutionMM module.
  local weight = torch.Tensor(1, nInputPlane, self.kernel:size(1),
                              self.kernel:size(2))
  for i = 1, self.nInputPlane do
    weight[1][i]:copy(self.kernel)
  end
  self.modules[MEAN_ESTIMATOR].modules[2].weight:copy(weight)
  self.modules[MEAN_ESTIMATOR].modules[2].bias:zero()

  -- other operation.
  self.modules[SUBTRACTOR] = nn.CSubTable()
  self.modules[DIVIDER] = nn.CDivTable()

  -- coefficient array, to adjust side effects.
  self.coef = torch.Tensor(1, 1, 1, 1)
end

function SpatialSubtractiveNormalizationBatch:updateOutput(input)
  -- compute side coefficients.
  if (input:size(4) ~= self.coef:size(4)) or
     (input:size(3) ~= self.coef:size(3)) or
     (input:size(1) ~= self.coef:size(1)) then
    local ones = input.new():resizeAs(input):fill(1)
    self.coef = self.modules[MEAN_ESTIMATOR]:updateOutput(ones)
    self.coef = self.coef:clone()
  end

  -- compute mean.
  self.localSums = self.modules[MEAN_ESTIMATOR]:updateOutput(input)
  self.adjustedSums = self.modules[DIVIDER]:updateOutput{self.localSums,
                                                         self.coef}
  self.output = self.modules[SUBTRACTOR]:updateOutput{input, self.adjustedSums}

  return self.output
end

function SpatialSubtractiveNormalizationBatch:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()

  -- backprop through all modules.
  local gradSub =
      self.modules[SUBTRACTOR]:updateGradInput({input, self.adjustedSums},
                                               gradOutput)
  local gradDiv =
      self.modules[DIVIDER]:updateGradInput({self.localSums, self.coef},
                                            gradSub[2])
  self.gradInput:add(
      self.modules[MEAN_ESTIMATOR]:updateGradInput(input, gradDiv[1]))
  self.gradInput:add(gradSub[1])

  return self.gradInput
end

