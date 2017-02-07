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

local nn = require('nn')
local cudnn = require('cudnn')

local inPlaceReLU = true

function torch.addNonlinearity(mconf, input)
  local nonlin
  if mconf.nonlinType == 'relu6' then
    nonlin = nn.ReLU6(inPlaceReLU)
  elseif mconf.nonlinType == 'relu' then
    nonlin = nn.ReLU(inPlaceReLU)
  elseif mconf.nonlinType == 'sigmoid' then
    nonlin = nn.Sigmoid()
  else
    error('Bad mconf.nonlinType (' .. mconf.nonlinType .. ').')
  end
  print('Adding non-linearity: ' .. nonlin:__tostring() .. ' (inplace ' ..
        tostring(inPlaceReLU) .. ')')
  return nonlin(input)
end

function torch.addBN(mconf, input, ofeats)
  local bn
  if mconf.batchNormAffine then
    if not mconf.is3D then
      bn = cudnn.SpatialBatchNormalization(
          ofeats, mconf.batchNormEps, mconf.batchNormMom,
          mconf.batchNormAffine)
    else
      bn = cudnn.VolumetricBatchNormalization(
          ofeats, mconf.batchNormEps, mconf.batchNormMom,
          mconf.batchNormAffine)
    end
  else
    -- cudnn's batch norm does not support turning affine parameters off.
    if not mconf.is3D then
      bn = nn.SpatialBatchNormalization(
          ofeats, mconf.batchNormEps, mconf.batchNormMom,
          mconf.batchNormAffine)
    else
      bn = nn.VolumetricBatchNormalization(
          ofeats, mconf.batchNormEps, mconf.batchNormMom,
          mconf.batchNormAffine)
    end
  end
  print('Adding batch norm: ' .. bn:__tostring__())
  return bn(input)
end

-- @param interFeats - feature size of the intermediate layers when
-- creating a low rank approximation.
function torch.addConv(mconf, input, ifeats, ofeats, k, up, rank, interFeats)
  if rank == nil then
    assert(interFeats == nil)
    if not mconf.is3D then
      rank = 2  -- Default is full rank.
    else
      rank = 3  -- Default is full rank.
    end
  end
  assert(math.fmod(k, 2) == 1, 'convolution size must be odd')
  local pad = (k - 1) / 2
  local conv
  if not mconf.is3D then
    if up > 1 then
      assert(rank == 2, 'Upsampling layers must be full rank')
      conv = nn.SpatialConvolutionUpsample(
          ifeats, ofeats, k, k, 1, 1, pad, pad, up, up)
      cudnn.convert(conv, cudnn)  -- Convert the inner convolution to cudnn.
    else
      if rank == 1 then
        conv = nn.Sequential()
        conv:add(cudnn.SpatialConvolution(
            ifeats, interFeats, k, 1, 1, 1, pad, 0))
        conv:add(cudnn.SpatialConvolution(
            interFeats, ofeats, 1, k, 1, 1, 0, pad))
      elseif rank == 2 then
        conv = cudnn.SpatialConvolution(ifeats, ofeats, k, k, 1, 1, pad, pad)
      else
        error('rank ' .. rank .. ' is invalid (1 or 2)')
      end
    end
  else
    if up > 1 then
      assert(rank == 3, 'Upsampling layers must be full rank')
      conv = nn.VolumetricConvolutionUpsample(
          ifeats, ofeats, k, k, k, 1, 1, 1, pad, pad, pad, up, up, up)
      cudnn.convert(conv, cudnn)  -- Convert the inner convolution to cudnn.
    else
      if rank == 1 then
        -- There are LOTS of ways of partitioning the 3D conv into a low rank
        -- approximation (order, number of features, etc).
        -- We're just going to arbitrarily choose just one low rank approx.
        conv = nn.Sequential()
        conv:add(cudnn.VolumetricConvolution(
            ifeats, interFeats, k, 1, 1, 1, 1, 1, pad, 0, 0))
        conv:add(cudnn.VolumetricConvolution(
            interFeats, interFeats, 1, k, 1, 1, 1, 1, 0, pad, 0))
        conv:add(cudnn.VolumetricConvolution(
            interFeats, ofeats, 1, 1, k, 1, 1, 1, 0, 0, pad))
      elseif rank == 2 then
        conv = nn.Sequential()
        conv:add(cudnn.VolumetricConvolution(
            ifeats, interFeats, k, k, 1, 1, 1, 1, pad, pad, 0))
        conv:add(cudnn.VolumetricConvolution(
            interFeats, ofeats, 1, k, k, 1, 1, 1, 0, pad, pad))
      elseif rank == 3 then
        conv = cudnn.VolumetricConvolution(
            ifeats, ofeats, k, k, k, 1, 1, 1, pad, pad, pad)
      else
        error('rank ' .. rank .. ' is invalid (1, 2 or 3)')
      end
    end
  end
  print('Adding convolution: ' .. conv:__tostring__())
  return conv(input), conv
end

function torch.addPooling(mconf, input, size)
  local pool
  if not mconf.is3D then
    if mconf.poolType == 'avg' then
      pool = cudnn.SpatialAveragePooling(size, size, size, size)
    elseif mconf.poolType == 'max' then
      pool = cudnn.SpatialMaxPooling(size, size, size, size)
    else
      error("Bad pool type.  Must be 'avg' or 'max'")
    end
  else
    if mconf.poolType == 'avg' then
      pool = cudnn.VolumetricAveragePooling(size, size, size, size, size, size)
    elseif mconf.poolType == 'max' then
      pool = cudnn.VolumetricMaxPooling(size, size, size, size, size, size)
    else
      error("Bad pool type.  Must be 'avg' or 'max'")
    end
  end
  if pool.__tostring__ == nil then
    print('Adding pooling: ' .. torch.type(pool))
  else
    print('Adding pooling: ' .. pool:__tostring__())
  end
  return pool(input)
end

function torch.checkYangSettings(mconf)
  if mconf.nonlinType ~= 'sigmoid' then
    error('ERROR: yang model must use nonlinType == "sigmoid"')
  end
  if not mconf.inputChannels.pDiv then
    error('ERROR: yang model must have pDiv input')
  end
  if not mconf.inputChannels.div then
    error('ERROR: yang model must have div input')
  end
  if mconf.inputChannels.UDiv then
    error('ERROR: yang model must not have UDiv input')
  end
  if not mconf.inputChannels.flags then
    error('ERROR: yang model must have flags input')
  end
end

