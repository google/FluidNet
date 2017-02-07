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

-- Top level training and validation script for FluidNet.
--
-- Usage:
-- Global options can be set from the command line, ie:
-- >> qlua fluid_net_train.lua -gpu 1 -train_preturb.rotation 20 
--
-- To print a list of options (and their defaults) use:
-- >> qlua fluid_net_train.lua -help

dofile('lib/include.lua')
local cudnn = torch.loadPackageSafe('cudnn')
local cutorch = torch.loadPackageSafe('cutorch')
local paths = require('paths')
local optim = require('optim')
local mattorch = torch.loadPackageSafe('mattorch')
local gnuplot = torch.loadPackageSafe('gnuplot')

-- ****************************** Define Config ********************************
local conf = torch.defaultConf()  -- Table with configuration and model params.
conf = torch.parseArgs(conf)  -- Overwrite conf params from the command line.
torch.makeGlobal('_conf', conf)
conf.modelDirname = conf.modelDir .. '/' .. conf.modelFilename

-- ****************************** Select the GPU *******************************
cutorch.setDevice(conf.gpu)
print("GPU That will be used:")
print(cutorch.getDeviceProperties(conf.gpu))

-- **************************** Load data from Disk ****************************
local tr = torch.loadSet(conf, 'tr') --Instance of DataBinary
torch.makeGlobal('_tr', tr)
local te = torch.loadSet(conf, 'te') --Instance of DataBinary
torch.makeGlobal('_te', te)

-- ***************************** Create the model ******************************
local mconf, model
if conf.loadModel then
  local mpath = conf.modelDirname
  if conf.resumeTraining then
    mpath = mpath .. '_lastEpoch'
  end
  print('Loading model from ' .. mpath)
  mconf, model = torch.loadModel(mpath)
  conf.newModel = nil
else
  assert(not conf.resumeTraining,
         'Cant resume training without loading a model!')
  model, mconf = torch.defineModel(conf, tr) -- in model.lua
  model:cuda()
  -- Visualize the model to file.
  -- graph.dot(model.fg, 'Forward Graph', conf.modelDirname .. '_fg')
  -- graph.dot(model.bg, 'Backward Graph', conf.modelDirname .. '_bg')
end
torch.makeGlobal('_mconf', mconf)
torch.makeGlobal('_model', model)

-- ********************* Define Criterion (loss) function **********************
print '==> defining loss function'
local criterion
if mconf.lossFunc == 'fluid' then
  criterion = nn.FluidCriterion(
      mconf.lossPLambda, mconf.lossULambda, mconf.lossDivLambda,
      mconf.lossFuncBorderWeight, mconf.lossFuncBorderWidth)
else
  error('Incorrect lossFunc value.')
end

criterion.sizeAverage = true
torch.makeGlobal('_criterion', criterion)
criterion:cuda()
print('    using criterion ' .. criterion:__tostring())

-- ***************************** Get the parameters ****************************
print '==> Extracting model parameters'
local parameters, gradParameters = model:getParameters()
torch.makeGlobal('_parameters', parameters)
torch.makeGlobal('_gradParameters', gradParameters)
collectgarbage()

-- *************************** Define the optimizer ****************************
print '==> Defining Optimizer'
local optimMethod
if mconf.optimizationMethod == 'sgd' then
  print("    Using SGD...")
  optimMethod = optim.sgd
elseif mconf.optimizationMethod == 'adam' then
  print("    Using ADAM...")
  optimMethod = optim.adam
elseif mconf.optimizationMethod == 'rmsprop' then
  print("    Using rmsprop...")
  optimMethod = optim.rmsprop
else
  print("    Using SGD...")
  optimMethod = optim.sgd
  mconf.optimizationMethod = "default-sgd"
end

-- ************************ Visualize a Training Batch *************************
--[[
_tr:visualizeBatch(_conf, _mconf)  -- Visualize random batch.
_tr:visualizeBatch(_conf, _mconf, {1})  -- Explicitly define batch samples.
--]]

-- *********************** Calculate dataset statistics ************************
-- Calculate some statistics about the input channels to the network.
--[[
trMean, trStd, trL2 = _tr:calcDataStatistics(_conf, _mconf)
_tr:plotDataStatistics(trMean, trStd, trL2)
teMean, teStd, teL2 = _te:calcDataStatistics(_conf, _mconf)
_te:plotDataStatistics(teMean, teStd, teL2)
--]]

-- ************************ Profile the model for the paper ********************
if conf.profile then
  local res = 128  -- The 3D data is 64x64x64 (which isn't that interesting).
  local profileTime = 10
  print('==> Profiling FPROP for ' ..  profileTime .. ' seconds' ..
        ' with grid res ' .. res)
  local nuchan, zdim
  if not mconf.is3D then
    nuchan = 2
    zdim = 1
  else
    nuchan = 3
    zdim = res
  end
  -- Create a minimal (empty) batch to do a few FPROPs.
  local batchGPU = {
     pDiv = torch.CudaTensor(1, 1, zdim, res, res):fill(0), 
     UDiv = torch.CudaTensor(1, nuchan, zdim, res, res):fill(0),
     flags = tfluids.emptyDomain(torch.CudaTensor(1, 1, zdim, res, res),
                                 mconf.is3D)
  }
  model:evaluate()  -- Turn off training (so batch norm doesn't get messed up).
  local input = torch.getModelInput(batchGPU)
  model:forward(input)  -- Input once before we start profiling.
  cutorch.synchronize()  -- Make sure everything is allocated fully.
  sys.tic()
  local niters = 0
  while sys.toc() < profileTime do
    model:forward(input)
    niters = niters + 1
  end
  cutorch.synchronize()  -- Flush the GPU buffer.
  local fpropTime = sys.toc() / niters
  print('    FPROP Time: ' .. 1000 * fpropTime .. ' ms / sample')

  -- Also calculate the total FLOPS (with a print out per layer).
  local verbose = 1  -- Print out only nn nodes (ignore graph containers).
  local flops, peakMemory = torch.CalculateFlops(model, input, verbose)
  print('    TOTAL FLOPS: ' .. torch.HumanReadableNumber(flops) .. 'flops')

  -- Store the flops in case we want it for later.
  mconf.flops = flops
  mconf.peakMemory = peakMemory
  mconf.fpropTime = fpropTime

  torch.cleanupModel(model)
end

-- ******************************* Training Loop *******************************
if conf.train then
  torch.mkdir(conf.modelDir)

  -- Saving parameters
  dofile("lib/save_parameters.lua")
  print '==> Saving parameters (mconf, conf)'
  torch.save(conf.modelDirname .. '_mconf.bin', mconf)
  torch.save(conf.modelDirname .. '_conf.bin', conf)
  -- saveParameters dumps conf and mconf to a human readable test file.
  torch.saveParameters(conf, mconf)

  local logger = torch.Logger(conf.modelDirname .. '_log.txt',
                              conf.resumeTraining)
  logger:setNames{'trLoss', 'trPLoss', 'trULoss', 'trDivLoss',
                  'trLongTermDivLoss', 'teLoss', 'tePLoss', 'teULoss',
                  'teDivLoss', 'teLongTermDivLoss'}

  -- Perform training.
  print '==> starting training loop!'
  while mconf.epoch < conf.maxEpochs do
    mconf.epoch = mconf.epoch + 1
    local trPerf = torch.runEpoch(
        {data = tr, conf = conf, mconf = mconf, model = model,
         criterion = criterion, parameters = parameters,
         gradParameters = gradParameters, optimMethod = optimMethod,
         epochType = 'train'})
    local tePerf = torch.runEpoch(
        {data = te, conf = conf, mconf = mconf, model = model,
         criterion = criterion, parameters = parameters, epochType = 'test'})

    -- Save model to disk as last epoch.
    torch.cleanupModel(model)
    torch.saveModel(mconf, model, conf.modelDirname .. '_lastEpoch')

    -- Check if this is the best model so far and if so save to disk (this is
    -- effectively an early-out mechanism).
    if tePerf.loss < mconf.optimState.bestPerf then
      print(' ==> This is the best model so far. Saving to disk.')
      mconf.optimState.bestPerf = tePerf.loss
      torch.saveModel(mconf, model, conf.modelDirname)
    end

    -- Log the performance results.
    logger:add{
        trPerf.loss, trPerf.pLoss, trPerf.uLoss, trPerf.divLoss,
        trPerf.longTermDivLoss, tePerf.loss, tePerf.pLoss, tePerf.uLoss,
        tePerf.divLoss, tePerf.longTermDivLoss}
  end
end

-- ********************* Visualize some inputs and outputs *********************
-- Create a random batch, FPROP using it and visualize the results
--[[
local samplenum = math.max(tr:nsamples() / 2,1)
local err, pred, batchCPU, batchGPU =
    torch.FPROPImage(conf, mconf, tr, model, criterion, {samplenum})
--]]

-- *************************** CALCULATE STATISTICS ****************************
-- First do a fast run-through of the test-set to measure test-set crit perf.
local tePerf = torch.runEpoch(
    {data = te, conf = conf, mconf = mconf, model = model,
     criterion = criterion, parameters = parameters, epochType = 'test'})
torch.save(conf.modelDirname .. '_tePerf.bin', tePerf)

-- Plot a histogram of the batch errors (sometimes it is helpful to debug bad
-- samples in the test-set, i.e. if Manta goes unstable).
--[[
local errs = {}
for _, indErrPair in pairs(tePerf.batchErr) do
  errs[#errs + 1] = indErrPair.err
end
gnuplot.hist(torch.FloatTensor(errs))
gnuplot.plot({'errs', torch.FloatTensor(torch.range(1, #errs)),
              torch.FloatTensor(errs), '-'})
--]]

-- Now do a more detailed analysis of the test and training sets (including
-- long term divergence prediction). This is quite slow.
local function CalcAndDumpStats(data, dataStr)
  local nSteps = 64  -- Use 128 for paper.
  local stats = torch.calcStats(
      {data = data, conf = conf, mconf = mconf, model = model, nSteps = nSteps})
  local fn = conf.modelDirname .. '_' .. dataStr .. 'Stats.bin'
  torch.save(fn, stats)
  print('Saved ' .. fn)
  if mattorch ~= nil then
    fn = conf.modelDirname .. '_' .. dataStr .. 'Stats.mat'
    matStats = stats.histData
    matStats['normDiv'] = stats.normDiv
    mattorch.save(fn, matStats)
    print('Saved ' .. fn)
  end
end
CalcAndDumpStats(te, 'te')
CalcAndDumpStats(tr, 'tr')

print('ALL DONE!')
