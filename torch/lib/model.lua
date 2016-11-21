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
-- For ReLU6 you'll have to clone my fork of nn and cunn (relu6 branch).
local nonlinType = 'relu'  -- Choices are: 'relu', 'relu6', 'sigmoid'.

-- This is the step size for the FEM approximation of gradP. We will verify
-- this step size against the training data during model instantiation.
local stepSizeX = 1
local stepSizeY = 1
local stepSizeZ = 1

-- Construct a dummy model that just copies the last frame's pressure and
-- outputs that. Used for a sanity check.
function torch.defineBaselineGraph(conf, mconf, data)
  assert(mconf.twoDim, 'Only 2D supported for now.')
  local input = nn.Identity()():annotate{name = 'input'}
  local pDiv = nn.SelectTable(1)(input):annotate{name = 'pDiv'}
  local UDiv = nn.SelectTable(2)(input):annotate{name = 'UDiv'}
  local p = nn.Identity()(pDiv)
  local gradPNet = nn.Sequential()
  gradPNet:add(nn.SpatialFiniteElements(stepSizeX, stepSizeY))
  gradPNet:add(nn.Squeeze())  -- output sz (batch x 2 x ydim x xdim)
  local gradP = gradPNet(p):annotate{name = 'gradP'}
  local U = nn.CSubTable()({UDiv, gradP}):annotate{name = 'U'}
  local inputNodes = {input}
  local outputNodes = {p, U}
  local model = nn.gModule(inputNodes, outputNodes)
  return model
end

function torch.defineModelGraph(conf, mconf, data)
  local inDims
  if not mconf.inputPToPModel then
    if mconf.twoDim then
      inDims = 3 -- 3D because of U (2D) and geom (1D)
    else
      inDims = 4 -- 4D because of U (3D) and geom (1D)
    end
  else
    if mconf.twoDim then
      inDims = 4 -- 4D because of p (1D) and U (2D) and geom (1D)
    else
      inDims = 5 -- 5D because of p (1D) and U (3D) and geom (1D)
    end 
  end

  -- Ideally, we should have explicit input nodes for pDiv, UDiv and geom.
  -- However, if one of these nodes does not terminate (i.e. pDiv may or may
  -- not be used), torch will throw an error.
  local input = nn.Identity()():annotate{name = 'input'}

  -- The model expects a table of {p, U, geom}
  local pDiv
  if mconf.inputPToPModel then
    pDiv = nn.SelectTable(1)(input):annotate{name = 'pDiv'}
  end
  local UDiv = nn.SelectTable(2)(input):annotate{name = 'UDiv'}
  local geom = nn.SelectTable(3)(input):annotate{name = 'geom'}

  if mconf.twoDim then
    -- Remove the unary z dimension from each of the input tensors.
    if pDiv ~= nil then
      pDiv = nn.Select(3, 1)(pDiv)
    end
    UDiv = nn.Select(3, 1)(UDiv)
    geom = nn.Select(3, 1)(geom)
  end

  local pModelInput
  if mconf.inputPToPModel then
    pModelInput = nn.JoinTable(2)({pDiv, UDiv, geom})
  else
    pModelInput = nn.JoinTable(2)({UDiv, geom})
  end
  pModelInput:annotate{name = 'pModelInput'}

  -- Some helper functions.
  local function addNonlinearity(input)
    local nonlin
    if nonlinType == 'relu6' then
      nonlin = nn.ReLU6(inPlaceReLU)
    elseif nonlinType == 'relu' then
      nonlin = nn.ReLU(inPlaceReLU)
    elseif nonlinType == 'sigmoid' then
      nonlin = nn.Sigmoid()
    else
      error('Bad nonlinType.')
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
  local function addConv(input, ifeats, ofeats, k, up)
    assert(math.fmod(k, 2) == 1, 'convolution size must be odd')
    local pad = (k - 1) / 2
    local conv
    if mconf.twoDim then
      if up > 1 then
        conv = nn.SpatialConvolutionUpsample(
            ifeats, ofeats, k, k, 1, 1, pad, pad, up, up)
        cudnn.convert(conv, cudnn)  -- Convert the inner convolution to cudnn.
      else
        conv = cudnn.SpatialConvolution(ifeats, ofeats, k, k, 1, 1, pad, pad)
      end
    else
      if up > 1 then
        conv = nn.VolumetricConvolutionUpsample(
            ifeats, ofeats, k, k, k, 1, 1, 1, pad, pad, pad, up, up, up)
        cudnn.convert(conv, cudnn)  -- Convert the inner convolution to cudnn.
      else
        conv = cudnn.VolumetricConvolution(ifeats, ofeats, k, k, k, 1, 1, 1,
                                           pad, pad, pad)
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
 
  local osize, ksize, psize, usize, lastLayerKSize, lastLayerUSize

  if mconf.twoDim then
    -- Small model.
    osize = {16, 32, 32, 64, 64, 32}  -- Conv # output features.
    ksize = {5, 5, 5, 5, 1, 1}  -- Conv filter size.
    psize = {2, 1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
    usize = {1, 1, 1, 1, 1, 1}  -- upsampling size (1 == no upsampling).

    -- The last layer is somewhat special (since the number of osize is always
    -- 1, we don't ever do pooling and we may or may not concatenate the input
    -- tensor with the input pressure).
    lastLayerKSize = 3
    lastLayerUSize = 2

    -- Note: upsampling is done WITHIN the pooling layer (using
    -- SpatialConvolutionUpsampling or VolumetricConvolutionUpsampling).
    -- Therefore you "can" have both upsampling AND pooling in the same layer,
    -- however this would be silly and so I assert against it (because it's
    -- probably a mistake).

    -- Large model.
    --[[
    osize = {16, 32, 32, 64, 64, 64}  -- Conv # output features.
    ksize = {7, 7, 5, 5, 1, 1}  -- Conv filter size.
    psize = {2, 1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
    usize = {1, 1, 1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
    lastLayerKSize = 3
    lastLayerUSize = 2
    --]]
  else
    -- Full (slow) model.
    osize = {16, 16, 32, 64, 32, 32}
    ksize = {5, 5, 3, 3, 1, 1}
    psize = {2, 1, 1, 1, 1, 1}
    usize = {1, 1, 1, 1, 1, 1}
    lastLayerKSize = 3
    lastLayerUSize = 2
    -- 30fps model.
    --[[
    osize = {16, 16, 16, 16, 32, 32}
    ksize = {3, 3, 3, 3, 1, 1}
    psize = {2, 2, 1, 1, 1, 1}
    usize = {1, 1, 1, 1, 1, 2}
    lastLayerKSize = 3
    lastLayerUSize = 2
    --]]
  end
  
  assert(#osize == #ksize and #osize >= 1)
  for lid = 1, #osize do
    if psize[lid] > 1 then
      assert(usize[lid] == 1, 'Pooling and upsampling in the same layer!')
    end
    hl = addConv(hl, inDims, osize[lid], ksize[lid], usize[lid])
    hl = addNonlinearity(hl)
    if psize[lid] > 1 then
      hl = addPooling(hl, psize[lid])
    end
    if mconf.addBatchNorm then
      hl = addBN(hl, osize[lid])
    end
    inDims = osize[lid]
  end

  if mconf.addPressureSkip and mconf.inputPToPModel then
    -- Concatenate the input pressure with the current hidden layer.
    hl = nn.JoinTable(2)({hl, pDiv})
    inDims = inDims + 1
  end

  -- Output pressure (1 slice): final conv layer.
  p = addConv(hl, inDims, 1, lastLayerKSize, lastLayerUSize)

  -- Final output nodes.
  p:annotate{name = 'p'}

  -- Construct a network to calculate the gradient of pressure.
  local matchManta = false
  local deltaUNet = nn.Sequential()
  if mconf.twoDim then
    deltaUNet:add(nn.VelocityUpdate(matchManta))
    deltaUNet:add(nn.Select(3, 1, 1))  -- Remove the unary z dimension.
  else
    -- We need to remove the unary feature dimension from each of the inputs.
    local par = nn.ParallelTable()
    par:add(nn.Select(2, 1, 1))  -- Remove from p.
    par:add(nn.Select(2, 1, 1))  -- Remove from geom.
    deltaUNet:add(par)
    deltaUNet:add(nn.VelocityUpdate(matchManta))
  end
  local deltaU = deltaUNet({p, geom}):annotate{name = 'deltaU'}

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
  local model
  if not mconf.baselineModel then
    model = torch.defineModelGraph(conf, mconf, data)
  else
    model = torch.defineBaselineGraph(conf, mconf, data)
  end

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
  assert(torch.type(target) == 'table' and #target == 2)
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

function torch.FPROPImageToDisk(conf, mconf, data, model, criterion, imgList, 
                                filenamePrefix)
  local oldBatchSize = conf.batchSize
  conf.batchSize = #imgList
  local batchCPU, batchGPU = data:visualizeBatch(conf, mconf, imgList, 1, true,
                                                 "_input_" .. filenamePrefix)
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
  data:_visualizeBatchData(outputData, 'predicted output', 1, true,
                           '_predicted_' ..filenamePrefix, conf, mconf)
  return err, pred, batchCPU, batchGPU
end

