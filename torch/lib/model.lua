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
    if mconf.twoDim then
      inDims = inDims + 2
    else
      inDims = inDims + 3
    end
  end
  if mconf.inputChannels.geom then
    inDims = inDims + 1
  else
    error('Are you sure you dont want geometry on input?')
  end
  if mconf.inputChannels.div then
    inDims = inDims + 1
  end

  print('Number of input channels: ' .. inDims)

  assert((mconf.inputChannels.div or mconf.inputChannels.pDiv or 
          mconf.inputChannels.UDiv),
         'Are you sure you dont want any non-geom field info?')

  -- Ideally, we should have explicit input nodes for pDiv, UDiv and geom.
  -- However, if one of these nodes does not terminate (i.e. pDiv may or may
  -- not be used), torch will throw an error.
  local input = nn.Identity()():annotate{name = 'input'}

  -- The model expects a table of {p, U, geom}. We ALWAYS get these channels
  -- from the input even if the model does not use them. This way we can
  -- define a constant API even as the user turns on and off input channels.
  local pDiv, UDiv, geom, div

  if mconf.inputChannels.pDiv or mconf.addPressureSkip then  
    pDiv = nn.SelectTable(1)(input):annotate{name = 'pDiv'}
  end

  if mconf.inputChannels.UDiv or mconf.inputChannels.div then
    UDiv = nn.SelectTable(2)(input):annotate{name = 'UDiv'}
  end

  if mconf.inputChannels.geom or mconf.inputChannels.div then
    geom = nn.SelectTable(3)(input):annotate{name = 'geom'}
  end
  
  if mconf.inputChannels.div then
    local geomScalar = nn.Select(2, 1)(geom)  -- Remove the feat dim
    local divScalar = nn.VelocityDivergence():cuda()({UDiv, geomScalar})
    -- Note: Div is now of size (batch x depth x height x width). We need to
    -- add back the unary feature dimension.
    div = nn.Unsqueeze(2)(divScalar)

    div:annotate{name = 'div'}
  end

  if mconf.twoDim then
    -- Remove the unary z dimension from each of the input tensors.
    if pDiv ~= nil then
      pDiv = nn.Select(3, 1)(pDiv)
    end
    if UDiv ~= nil then
      UDiv = nn.Select(3, 1)(UDiv)
    end
    if geom ~= nil then
      geom = nn.Select(3, 1)(geom)
    end
    if div ~= nil then
      div = nn.Select(3, 1)(div)
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
  if mconf.inputChannels.geom then
    inputChannels[#inputChannels + 1] = geom
  end

  local pModelInput = nn.JoinTable(2)(inputChannels)
  pModelInput:annotate{name = 'pModelInput'}

  -- Some helper functions.
  local function addNonlinearity(input)
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
  local function addBN(input, ofeats)
    local bn
    if mconf.batchNormAffine then
      if mconf.twoDim then
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
      if mconf.twoDim then
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
  local function addConv(input, ifeats, ofeats, k, up, rank, interFeats)
    if rank == nil then
      assert(interFeats == nil)
      if mconf.twoDim then
        rank = 2  -- Default is full rank.
      else
        rank = 3  -- Default is full rank.
      end
    end
    assert(math.fmod(k, 2) == 1, 'convolution size must be odd')
    local pad = (k - 1) / 2
    local conv
    if mconf.twoDim then
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
    return conv(input)
  end
  local function addPooling(input, size)
    local pool
    if mconf.twoDim then
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

  -- Construct a network that takes in pModelInput and outputs
  -- divergence free pressure.
  local p
  local hl = pModelInput
 
  local osize, ksize, psize, usize, rank
  local interFeats

  local function checkYangSettings(mconf)
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
    if not mconf.inputChannels.geom then
      error('ERROR: yang model must have geom input')
    end
  end

  if mconf.twoDim then
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
      -- with p, divergence and geom as input.
      checkYangSettings(mconf)
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
      checkYangSettings(mconf)
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
  if mconf.twoDim then
    assert(rank[#rank] == 2, 'Last layer must be full rank')
  else
    assert(rank[#rank] == 3, 'Last layer must be full rank')
  end

  print('Model type: ' .. mconf.modelType)
  
  assert(#osize == #ksize and #osize >= 1)
  for lid = 1, #osize - 1 do
    if psize[lid] > 1 then
      assert(usize[lid] == 1, 'Pooling and upsampling in the same layer!')
    end
    hl = addConv(hl, inDims, osize[lid], ksize[lid], usize[lid], rank[lid],
                 interFeats[lid])
    hl = addNonlinearity(hl)
    if psize[lid] > 1 then
      hl = addPooling(hl, psize[lid])
    end
    if mconf.addBatchNorm then
      hl = addBN(hl, osize[lid])
    end
    inDims = osize[lid]
  end

  if mconf.addPressureSkip then
    -- Concatenate the input pressure with the current hidden layer.
    hl = nn.JoinTable(2)({hl, pDiv})
    inDims = inDims + 1
  end

  -- Output pressure (1 slice): final conv layer. (full rank)
  p = addConv(hl, inDims, 1, ksize[#ksize], usize[#usize])

  -- Final output nodes.
  p:annotate{name = 'p'}

  -- Construct a network to calculate the gradient of pressure.
  local matchManta = false
  local deltaUNet = nn.VelocityUpdate(matchManta)
  -- Manta may have p in some unknown units.
  -- Therefore we should apply a constant scale to p in order to
  -- correct U (recall U = UDiv - grad(p)).
  local pScaled = nn.Mul()(p):annotate{name = 'pScaled'}
  local deltaU = deltaUNet({pScaled, geom}):annotate{name = 'deltaU'}
  local U = nn.CSubTable()({UDiv, deltaU}):annotate{name = 'U'}

  if mconf.twoDim then
    -- We need to add BACK a unary dimension.
    p = nn.Unsqueeze(3)(p)  -- Adds a new singleton dimension at dim 3.
    U = nn.Unsqueeze(3)(U)
  end

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
  mconf.twoDim = data.twoDim  -- The dimension of the model depends on the data.

  print('==> Creating model...')
  mconf.netDownsample = 1

  -- Start putting together the new model.
  local model = torch.defineModelGraph(conf, mconf, data)

  return model, mconf
end

function torch.getModelInput(batch)
  return {batch.pDiv, batch.UDiv, batch.geom}
end

function torch.parseModelInput(input)
  assert(torch.type(input) == 'table' and #input == 3)
  local pDiv = input[1]
  local UDiv = input[2]
  local geom = input[3]
  return pDiv, UDiv, geom
end

function torch.getModelTarget(batch)
  return {batch.pTarget, batch.UTarget, batch.geom}
end

function torch.parseModelTarget(target)
  assert(torch.type(target) == 'table' and #target == 3, 'Bad target!')
  local pTarget = target[1]
  local UTarget = target[2]
  local geom = target[3]
  return pTarget, UTarget, geom
end

function torch.parseModelOutput(output)
  assert(torch.type(output) == 'table' and #output == 2)
  local pPred = output[1]
  local UPred = output[2]
  return pPred, UPred
end

-- @param normVal: from batchGPU/CPU.normVal.
-- NO LONGER USED BUT LEFT FOR REFERENCE.
function torch.undoNormalizeModelOutput(conf, mconf, output, normVal)
  local pPred, UPred = torch.parseModelOutput(output)
  pPred:cmul(normVal:expandAs(pPred))
  UPred:cmul(normVal:expandAs(UPred))
end

-- @param normVal: from batchGPU/CPU.normVal.
-- NO LONGER USED BUT LEFT FOR REFERENCE.
function torch.normalizeModelInput(conf, mconf, input, normVal)
  local pDiv, UDiv, geom = torch.parseModelInput(input)
  pDiv:cdiv(normVal:expandAs(pDiv))
  UDiv:cdiv(normVal:expandAs(UDiv))
  -- Note: we don't normalize geometry.
end

-- @param normVal: from batchGPU/CPU.normVal.
-- NO LONGER USED BUT LEFT FOR REFERENCE.
function torch.undoNormalizeModelTarget(conf, mconf, target, normVal)
  local pTarget, UTarget = torch.parseModelTarget(target)
  pTarget:cmul(normVal:expandAs(pTarget))
  UTarget:cmul(normVal:expandAs(UTarget))
end

-- @param normVal: from batchGPU/CPU.normVal.
-- NO LONGER USED BUT LEFT FOR REFERENCE.
function torch.normalizeModelTarget(conf, mconf, target, normVal)
  local pTarget, UTarget = torch.parseModelTarget(target)
  pTarget:cdiv(normVal:expandAs(pTarget))
  UTarget:cdiv(normVal:expandAs(UTarget))
end

-- Calculate the scale for input normalization (global per sample). 
-- Basically, we we are solving a linear system Ax = b, so we have the 
-- freedom to (per batch sample) scale the input and outputs by any constant
-- that we want.
-- NO LONGER USED BUT LEFT FOR REFERENCE.
function torch.calcNormValue(conf, mconf, UDiv, normVal)
  -- Flatten UDiv to a 2D tensor of size (batch, depth x width x height).
  local nbatch = UDiv:size(1)
  UDiv = UDiv:view(nbatch, UDiv:numel() / nbatch)
  normVal = normVal:view(nbatch, 1)
  torch.std(normVal, UDiv, 2)  -- STD along dim 2 of UDiv (result in normVal).
  normVal:add(1e-6)  -- To cover precision errors.
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
      geom = batchGPU.geom
  }
  data:_visualizeBatchData(outputData, 'predicted output')
  return err, pred, batchCPU, batchGPU
end

