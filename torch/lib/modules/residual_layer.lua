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

-- A container class for a basic residual layer.
--
-- The implementation here is the implementation from Deep Residual Learning
-- for Image Recognition, He et al. (a good diagram can be found here:
-- http://torch.ch/blog/2016/02/04/resnets.html - "reference paper" version).

local nn = require('nn')
local cudnn = require('cudnn')

local ResidualLayer, parent = torch.class('nn.ResidualLayer', 'nn.Container')

-- Constructor. Note: convolutions in this layer have dW and dH == 1.
-- Additionally, padding is always added to prevent pixel loss.
--
-- @param nInputPlane - The number of input planes.
-- @param nOutputPlaneStage1 - The number of output planes in the first conv
-- layer. This is a hidden layer and is completely internal.
-- @param nOutputPlane - The number of output planes.
-- @param kW/kH - The filter size.
-- @param addBatchNorm - If true then batch normalization is added (probably
-- a good idea to use it).
-- @param idenityShortcut - If true then an identity is used for the skip
-- connection (as described in the paper). In the case of nInputPlane >
-- nOutputPlane, the tensor will be truncated. If nInputPlane < nOutputPlane,
-- the tensor will be zero padded. If false then a learned 1x1 will up or down
-- scale the dimension.
-- @param batchNormEps, batchNormMom, batchNormAffine - OPTIONAL bnorm params.
function ResidualLayer:__init(nInputPlane, nOutputPlaneStage1,
                              nOutputPlane, kW, kH, addBatchNorm,
                              identityShortcut, batchNormEps, batchNormMom,
                              batchNormAffine)
  assert(math.fmod(kW, 2) == 1 and math.fmod(kH, 2) == 1,
         'filter size must be odd!')
  self.nInputPlane = nInputPlane
  self.nOutputPlaneStage1 = nOutputPlaneStage1
  self.nOutputPlane = nOutputPlane
  self.kW = kW
  self.kH = kH
  self.dW = 1
  self.dH = 1
  self.padW = (kW - 1) / 2
  self.padH = (kH - 1) / 2
  assert(addBatchNorm ~= nil)  -- Otherwise it defaults to false.
  self.addBatchNorm = addBatchNorm
  assert(identityShortcut ~= nil)
  self.identityShortcut = identityShortcut
  self.batchNormEps = batchNormEps
  self.batchNormMom = batchNormMom
  self.batchNormAffine = batchNormAffine

  local net = nn.Sequential()
  local split = nn.ConcatTable()

  local branch1 = nn.Sequential()
  -- First Convolution.
  branch1:add(cudnn.SpatialConvolution(
      self.nInputPlane, self.nOutputPlaneStage1, self.kW, self.kH, self.dW,
      self.dH, self.padW, self.padH))
  -- Batch Norm (optional).
  if self.addBatchNorm then
    branch1:add(cudnn.SpatialBatchNormalization(
        self.nOutputPlaneStage1, self.batchNormEps, self.batchNormMom,
        self.batchNormAffine))
  end
  -- ReLU.
  branch1:add(cudnn.ReLU(true))  -- In-place.
  -- Second Convolution.
  branch1:add(cudnn.SpatialConvolution(
      self.nOutputPlaneStage1, self.nOutputPlane, self.kW, self.kH,
      self.dW, self.dH, self.padW, self.padH))
  -- Batch Norm (optional).
  if self.addBatchNorm then
    branch1:add(cudnn.SpatialBatchNormalization(
        self.nOutputPlane, self.batchNormEps, self.batchNormMom,
        self.batchNormAffine))
  end

  local branch2
  if nInputPlane == nOutputPlane then
    branch2 = nn.Identity()
  else
    if not self.identityShortcut then
      -- Scale feature dim using 1x1 convolution.
      branch2 = cudnn.SpatialConvolution(self.nInputPlane, self.nOutputPlane,
                                         1, 1, 1, 1, 0, 0)
      -- Batch Norm (optional).
      if self.addBatchNorm then
        branch1:add(cudnn.SpatialBatchNormalization(
            self.nOutputPlane, self.batchNormEps, self.batchNormMom,
            self.batchNormAffine))
      end
    else
      if self.nInputPlane < self.nOutputPlane then
        -- Upscale feature dim by Zero Padding.
        local featpad = self.nOutputPlane - self.nInputPlane
        local dim = 2
        local nInputDim = 4  -- (batch, feat, height, width)
        local value = 0
        branch2 = nn.Padding(dim, featpad, nInputDim, value)
      else
        -- Narrow feature dim.
        branch2 = nn.Narrow(2, 1, self.nOutputPlane)
      end
    end
  end

  split:add(branch1)
  split:add(branch2)
  net:add(split)
  -- Addition.
  net:add(nn.CAddTable())
  -- ReLU.
  net:add(cudnn.ReLU(true))  -- In-place.

  self.modules = {net}

  self.output = self.modules[1].output
  self.gradInput = self.modules[1].gradInput
end

function ResidualLayer:updateOutput(input)
  self.output = self.modules[1]:updateOutput(input)
  return self.output
end

function ResidualLayer:updateGradInput(input, gradOutput)
  self.gradInput = self.modules[1]:updateGradInput(input, gradOutput)
  return self.gradInput
end

function ResidualLayer:accGradParameters(input, gradOutput, scale)
  self.modules[1]:accGradParameters(input, gradOutput, scale)
end

function ResidualLayer:accUpdateGradParameters(input, gradOutput, lr)
  self.modules[1]:accUpdateGradParameters(input, gradOutput, lr)
end

function ResidualLayer:sharedAccUpdateGradParameters(input, gradOutput, lr)
  self.modules[1]:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function ResidualLayer:type(type)
  parent.type(self, type)
  for i = 1, #self.modules do
    self.modules[i]:type(type)
  end
  return self
end

function ResidualLayer:__tostring__()
  local s = string.format(
      '%s(%d -> %d -> %d, %dx%d, bn=%s, ident=%s)', torch.type(self),
      self.nInputPlane, self.nOutputPlaneStage1, self.nOutputPlane,
      self.kW, self.kH, tostring(self.addBatchNorm),
      tostring(self.identityShortcut))
  return s
end
