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
local tfluids = require('tfluids')

local inPlaceReLU = true

-- This is the step size for the FEM approximation of gradP. We will verify
-- this step size against the training data during model instantiation.
local stepSizeX = 1
local stepSizeY = 1
local stepSizeZ = 1

function torch.defineModelGraph(conf, mconf, data)
  local inDims = 0
  if mconf.inputChannels.pDiv then
    inDims = inDims + 1
  end
  if mconf.inputChannels.UDiv then
    if mconf.is3D then
      inDims = inDims + 3
    else
      inDims = inDims + 2
    end
  end
  if mconf.inputChannels.flags then
    inDims = inDims + 1
  else
    error('Are you sure you dont want flags on input?')
  end
  if mconf.inputChannels.div then
    inDims = inDims + 1
  end

  print('Number of input channels: ' .. inDims)

  assert((mconf.inputChannels.div or mconf.inputChannels.pDiv or 
          mconf.inputChannels.UDiv),
         'Are you sure you dont want any (U, div or p) fields?')

  -- Ideally, we should have explicit input nodes for pDiv, UDiv and flags.
  -- However, if one of these nodes does not terminate (i.e. pDiv may or may
  -- not be used), torch will throw an error.
  local input = nn.Identity()():annotate{name = 'input'}

  -- The model expects a table of {p, U, flags}. We ALWAYS get these channels
  -- from the input even if the model does not use them. This way we can
  -- define a constant API even as the user turns on and off input channels.
  local pDiv, UDiv, flags, div

  if (mconf.inputChannels.pDiv or mconf.addPressureSkip or
      (mconf.normalizeInput and mconf.normalizeInputChan == 'pDiv')) then  
    pDiv = nn.SelectTable(1)(input):annotate{name = 'pDiv'}
  end

  if (mconf.inputChannels.UDiv or mconf.inputChannels.div or
      (mconf.normalizeInput and mconf.normalizeInputChan == 'uDiv')) then
    UDiv = nn.SelectTable(2)(input):annotate{name = 'UDiv'}
  end

  if mconf.inputChannels.flags or mconf.inputChannels.div then
    flags = nn.SelectTable(3)(input):annotate{name = 'flags'}
  end
  -- There's not really a good reason not to include flags, so lets
  -- make sure this doesn't happen.
  assert(mconf.inputChannels.flags == true, 'Are you sure you dont want flags?')
 
  if UDiv ~= nil then
    -- Apply setWallBcs to zero out obstacle velocities on the boundary.
    UDiv = tfluids.SetWallBcs()({UDiv, flags})
  end
 
  if mconf.inputChannels.div then
    div = tfluids.VelocityDivergence():cuda()({UDiv, flags})
    div:annotate{name = 'div'}
  end

  local scale
  if mconf.normalizeInput then
    local scaleNet = nn.Sequential()
    -- reshape from (b x 2/3 x d x h x w) to (b x -1)
    scaleNet:add(nn.View(-1):setNumInputDims(4))
    if mconf.normalizeInputFunc == 'std' then
      scaleNet:add(nn.StandardDeviation(2))  -- output is size (b x 1)
    elseif mconf.normalizeInputFunc == 'norm' then
      scaleNet:add(nn.Power(2))
      scaleNet:add(nn.Sum(2))  -- output is size (b)
      scaleNet:add(nn.Unsqueeze(2, 1))  -- output is size (b x 1)
      scaleNet:add(nn.Sqrt())
    else
      error('Incorrect normalize input function')
    end
    scaleNet:add(nn.Clamp(mconf.normalizeInputThrehsold, math.huge))

    if mconf.normalizeInputChan == 'UDiv' then
      scale = scaleNet(UDiv)
    elseif mconf.normalizeInputChan == 'pDiv' then
      scale = scaleNet(pDiv)
    elseif mconf.normalizeInputChan == 'div' then
      scale = scaleNet(div)
    else
      error('Incorrect normalize input channel.')
    end
    scale:annotate{name = 'input_scale'}

    if pDiv ~= nil then
      pDiv = nn.ApplyScale(true)({pDiv, scale})  -- Applies pDiv *= (1 / scale)
      pDiv:annotate{name = 'pDivScaled'}
    end
    if UDiv ~= nil then
      UDiv = nn.ApplyScale(true)({UDiv, scale})
      UDiv:annotate{name = 'UDivScaled'}
    end
    if div ~= nil then
      div = nn.ApplyScale(true)({div, scale})
      div:annotate{name = 'divScaled'}
    end
  end

  local pModelInput
  local inputChannels = {}
  if mconf.inputChannels.pDiv then
    inputChannels[#inputChannels + 1] = pDiv
  end
  if mconf.inputChannels.UDiv then
    inputChannels[#inputChannels + 1] = UDiv
  end
  if mconf.inputChannels.div then
    inputChannels[#inputChannels + 1] = div
  end
  if mconf.inputChannels.flags then
    -- FlagsToOccupancy will create a [0, 1] grid out of the Manta flags.
    inputChannels[#inputChannels + 1] = tfluids.FlagsToOccupancy()(flags)
  end

  local pModelInput = nn.JoinTable(2)(inputChannels)
  pModelInput:annotate{name = 'pModelInput'}

  if not mconf.is3D then
    -- Remove the unary z dimension from the input for SpatialConvolution.
    pModelInput = nn.Select(3, 1)(pModelInput)
  end

  -- Construct a network that takes in pModelInput and outputs
  -- divergence free pressure.
  local p
 
  local osize, ksize, psize, usize, rank
  local interFeats

  if not mconf.is3D then
    if mconf.modelType == 'tog' then
      -- Small model.
      osize = {16, 32, 32, 64, 64, 32, 1}  -- Conv # output features.
      ksize = {5, 5, 5, 5, 1, 1, 3}  -- Conv filter size.
      psize = {2, 1, 1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1, 1, 1, 2}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2, 2, 2, 2}
      interFeats = {nil, nil, nil, nil, nil, nil, nil}

      -- Note: upsampling is done WITHIN the conv layer (using
      -- SpatialConvolutionUpsampling or VolumetricConvolutionUpsampling).
      -- Therefore you "can" have both upsampling AND pooling in the same layer,
      -- however this would be silly and so I assert against it (because it's
      -- probably a mistake).
    elseif mconf.modelType == 'default' then
      osize = {16, 16, 16, 16, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2, 2}
      interFeats = {nil, nil, nil, nil, nil}
    elseif mconf.modelType == 'yang' then
      -- From the paper: Data-driven projection method in fluid
      -- simulation, Yang et al., 2016.
      -- "In this paper, the neural network has three hidden layers and six
      -- neurons per hidden layer."      

      -- Note: the Yang paper defines a per-patch network, however this can
      -- be exactly mplemented as a "fully-convolutional" network with 1x1x1
      -- stages for the remaining hidden convolution layers.
      -- They also use only the surrounding neighbor pixels as input context,
      -- with p, divergence and flags as input.
      torch.checkYangSettings(mconf)
      osize = {6, 6, 6, 1}
      ksize = {3, 1, 1, 1}  -- They define a per patch network, whic
      psize = {1, 1, 1, 1}  -- They do not pool or upsample
      usize = {1, 1, 1, 1}  -- They do not pool or upsample
      rank = {2, 2, 2, 2}  -- Always full rank.
      interFeats = {nil, nil, nil, nil}
    else
      error('Incorrect modelType for 2D model.')
    end
  else
    if mconf.modelType == 'tog' then
      -- Fast model.
      osize = {16, 16, 16, 16, 32, 32, 1}
      ksize = {3, 3, 3, 3, 1, 1, 3}
      psize = {2, 2, 1, 1, 1, 1, 1}
      usize = {1, 1, 1, 1, 1, 2, 2}
      rank = {3, 3, 3, 3, 3, 3, 3}
      interFeats = {nil, nil, nil, nil, nil, nil, nil}
    elseif mconf.modelType == 'default' then
      osize = {8, 8, 8, 8, 1}
      ksize = {3, 3, 3, 1, 1}
      psize = {1, 1, 1, 1, 1}
      usize = {1, 1, 1, 1, 1}
      rank = {3, 3, 3, 3, 3}
      interFeats = {nil, nil, nil, nil, nil}
    elseif mconf.modelType == 'yang' then
      torch.checkYangSettings(mconf)
      osize = {6, 6, 6, 1}
      ksize = {3, 1, 1, 1}
      psize = {1, 1, 1, 1}
      usize = {1, 1, 1, 1}
      rank = {3, 3, 3, 3}
      interFeats = {nil, nil, nil, nil}
   else
     error('Incorrect modelType for 3D model.')
   end
  end

  -- NOTE: The last layer is somewhat special (since the number of osize is
  -- always 1, we don't ever do pooling and we may or may not concatenate
  -- the input tensor with the input pressure).
  assert(osize[#osize] == 1, 'Last layer osize must be 1 (pressure)')
  assert(psize[#psize] == 1, 'Pooling is not allowed in the last layer')
  if not mconf.is3D then
    assert(rank[#rank] == 2, 'Last layer must be full rank')
  else
    assert(rank[#rank] == 3, 'Last layer must be full rank')
  end

  -- Check the multi-res parameters.
  assert(mconf.banksNum >= 1)
  assert(mconf.banksSplitStage < mconf.banksJoinStage)
  -- For now (just to make our lives easy), make sure split is BEFORE the last
  -- layer.
  assert(mconf.banksSplitStage >= 1 and mconf.banksSplitStage < #osize)
  assert(mconf.banksJoinStage >= 1 and mconf.banksJoinStage < #osize)

  print('Model type: ' .. mconf.modelType)

  local hl = {pModelInput}  -- hidden layer for bank 'i'
 
  assert(#osize == #ksize and #osize >= 1)
  for lid = 1, #osize - 1 do
    if mconf.banksNum > 1 and lid == mconf.banksSplitStage then
      -- Split the hidden layer into a Gaussian pyramid.
      for ibank = 2, mconf.banksNum do
        local modDown
        if not mconf.is3D then
          modDown = nn.SpatialAveragePooling(2, 2, 2, 2)
        else
          modDown = nn.VolumetricAveragePooling(2, 2, 2, 2, 2, 2)
        end
        hl[ibank] = modDown(hl[ibank - 1]):annotate{
            name = 'Bank ' .. ibank .. ': downsample'}
      end
    end
    if mconf.banksNum > 1 and lid == mconf.banksJoinStage then
      -- Join the hidden layers together.
      -- First bring the banks into canonical resolution.
      for ibank = 2, mconf.banksNum do
        local ratio = math.pow(2, ibank - 1)
        if not mconf.is3D then
          hl[ibank] = nn.SpatialUpSamplingNearest(ratio)(hl[ibank])
        else
          hl[ibank] = tfluids.VolumetricUpSamplingNearest(ratio)(hl[ibank])
        end
        hl[ibank]:annotate{name = 'Bank ' .. ibank .. ': Upsample'}
      end
      -- Now aggregate the hidden layers.
      if mconf.banksAggregateMethod == 'concat' then
        hl = {nn.JoinTable(2)(hl)}
        hl[1]:annotate{name = 'Concat Feats'}
        inDims = inDims * mconf.banksNum
      elseif mconf.banksAggregateMethod == 'add' then
        hl = {nn.CAddTable()(hl)}
        hl[1]:annotate{name = 'Add Feats'}
      else
        error('ERROR: unsupported mconf.banksAggregateMethod')
      end
    end
    local conv
    for ibank = 1, #hl do
      print('Bank ' .. ibank .. ':')
      if psize[lid] > 1 then
        assert(usize[lid] == 1, 'Pooling and upsampling in the same layer!')
      end
      if mconf.banksWeightShare and ibank > 1 then
        local curConv = conv:clone('weight', 'bias', 'gradWeight', 'gradBias')
        print('Adding shared convolution: ' .. curConv:__tostring__())
        hl[ibank] = curConv(hl[ibank])
      else
        hl[ibank], conv = torch.addConv(
            mconf, hl[ibank], inDims, osize[lid], ksize[lid], usize[lid],
            rank[lid], interFeats[lid])
      end
      hl[ibank]:annotate{name = 'Bank ' .. ibank .. ': conv stage ' .. lid}
      hl[ibank] = torch.addNonlinearity(mconf, hl[ibank])
      hl[ibank]:annotate{name = 'Bank ' .. ibank .. ': non-linearity'}
      if psize[lid] > 1 then
        hl[ibank] = torch.addPooling(mconf, hl[ibank], psize[lid])
      end
      if mconf.addBatchNorm then
        hl[ibank] = torch.addBN(mconf, hl[ibank], osize[lid])
      end
    end
    inDims = osize[lid]
  end

  assert(#hl == 1, 'Last layer should get a single res bank')
  hl = hl[1]

  if mconf.addPressureSkip then
    -- Concatenate the input pressure with the current hidden layer.
    hl = nn.JoinTable(2)({hl, pDiv})
    inDims = inDims + 1
  end

  -- Output pressure (1 slice): final conv layer. (full rank)
  p = torch.addConv(mconf, hl, inDims, 1, ksize[#ksize], usize[#usize])

  if not mconf.is3D then
    -- We need to add back the unary dimension.
    p = nn.Unsqueeze(3)(p)  -- Adds a new singleton dimension at dim 3.
  end

  -- Final output nodes.
  p:annotate{name = 'pPred'} 

  -- Manta may have p in some unknown units.
  -- Therefore we should apply a constant scale to p in order to
  -- correct U (recall U = UDiv - grad(p)).
  -- local pScaled = nn.Mul()(p):annotate{name = 'pMul'}
  -- EDIT(tompson Jan 30 2017): no longer needed now we match Manta.  Needlessly
  -- adds a nullspace to the MSE error manifold (i.e. arbitrary scale).
  local U = tfluids.VelocityUpdate()({p, UDiv, flags})
  U:annotate{name = 'UPred'}

  -- Now we need to UNDO the scale factor we applied on the input.
  if mconf.normalizeInput then
    p = nn.ApplyScale(false)({p, scale})  -- Applies p' *= scale
    U = nn.ApplyScale(false)({U, scale})
  end

  -- Now (as we do in Manta) call setWallBcs again after velocity update.
  U = tfluids.SetWallBcs()({U, flags})

  p:annotate{name = 'p'}
  U:annotate{name = 'U'}

  -- Construct final graph.
  local inputNodes = {input}
  local outputNodes = {p, U}
  local model = nn.gModule(inputNodes, outputNodes)

  return model
end

function torch.defineModel(conf, data)
  -- Move the newModel parameters into mconf.
  local mconf = {}
  for key, value in pairs(conf.newModel) do
    mconf[key] = value
  end
  conf.newModel = nil
  mconf.epoch = 0
  mconf.is3D = data.is3D  -- The dimension of the model depends on the data.

  print('==> Creating model...')

  -- Start putting together the new model.
  local model = torch.defineModelGraph(conf, mconf, data)

  return model, mconf
end

function torch.getModelInput(batch)
  return {batch.pDiv, batch.UDiv, batch.flags}
end

function torch.parseModelInput(input)
  assert(torch.type(input) == 'table' and #input == 3)
  local pDiv = input[1]
  local UDiv = input[2]
  local flags = input[3]
  return pDiv, UDiv, flags
end

function torch.getModelTarget(batch)
  return {batch.pTarget, batch.UTarget, batch.flags}
end

function torch.parseModelTarget(target)
  assert(torch.type(target) == 'table' and #target == 3, 'Bad target!')
  local pTarget = target[1]
  local UTarget = target[2]
  local flags = target[3]
  return pTarget, UTarget, flags
end

function torch.parseModelOutput(output)
  assert(torch.type(output) == 'table' and #output == 2)
  local pPred = output[1]
  local UPred = output[2]
  return pPred, UPred
end

-- Resize the output, gradInput, etc temporary tensors to zero (so that the
-- on disk size is smaller).
function torch.cleanupModel(node)
  node:clearState()
  collectgarbage()
end

local function getMconfFilename(modelFilename)
  return modelFilename .. '_mconf.bin'
end

function torch.loadModel(modelFilename)
  assert(paths.filep(modelFilename), 'Could not find model: ' .. modelFilename)
  local mconfFilename = getMconfFilename(modelFilename)
  assert(paths.filep(mconfFilename), 'Could not find mconf: ' .. mconfFilename)
  local model = torch.load(modelFilename)
  local mconf = torch.load(mconfFilename)
  print('==> Loaded model ' .. modelFilename)
  return mconf, model
end

function torch.saveModel(mconf, model, modelFilename)
  torch.cleanupModel(model)  -- Massively reduces size on disk
  torch.save(modelFilename, model)
  torch.save(getMconfFilename(modelFilename), mconf)
  print('    - Model saved to: ' .. modelFilename)
end

function torch.FPROPImage(conf, mconf, data, model, criterion, imgList)
  local oldBatchSize = conf.batchSize
  conf.batchSize = #imgList
  local batchCPU, batchGPU = data:visualizeBatch(conf, mconf, imgList)
  conf.batchSize = oldBatchSize
  local input = torch.getModelInput(batchGPU)
  local target = torch.getModelTarget(batchGPU)
  local pred = model:forward(input)
  local err = criterion:forward(pred, target)

  local outputData = {
      p = pred[1],
      U = pred[2],
      flags = batchGPU.flags
  }
  data:_visualizeBatchData(outputData, 'predicted output')
  return err, pred, batchCPU, batchGPU
end

