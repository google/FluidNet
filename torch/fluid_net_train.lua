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
local ProFi = torch.loadPackageSafe('ProFi')
local mattorch = torch.loadPackageSafe('mattorch')

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

--[[
-- Funcs to visualize the data.
_tr:visualizeData(_conf)  -- visualize a random run.
_tr:visualizeData(_conf, 1) -- visualize a particular run.
--]]

-- ********************* Define Criterion (loss) function **********************
print '==> defining loss function'
local criterion
if mconf.lossFunc == 'fluid' then
  criterion = nn.FluidCriterion(
      mconf.lossPLambda, mconf.lossULambda, mconf.lossDivLambda,
      mconf.lossFuncScaleInvariant)
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
elseif mconf.optimizationMethod == 'adagrad' then
  print("    Using ADAGRAD...")
  optimMethod = optim.adagrad
elseif mconf.optimizationMethod == 'adamax' then
  print("    Using ADAMAX...")
  optimMethod = optim.adamax
elseif mconf.optimizationMethod == 'adadelta' then
  print("    Using ADADELTA...")
  optimMethod = optim.adadelta
elseif mconf.optimizationMethod == 'cg' then
  print("    Using Conjugate Gradient...")
  optimMethod = optim.cg
elseif mconf.optimizationMethod == 'rmsprop' then
  print("    Using rmsprop...")
  optimMethod = optim.rmsprop
elseif mconf.optimizationMethod == 'rprop' then
  print("    Using rprop...")
  optimMethod = optim.rprop
elseif mconf.optimizationMethod == 'lbfgs' then
  print("    Using lbfgs...")
  optimMethod = optim.lbfgs
else
  print("    Using SGD...")
  optimMethod = optim.sgd
  mconf.optimizationMethod = "default-sgd"
end

-- ************************ a Visualize Training Batch *************************
--[[
_tr:visualizeBatch(_conf, _mconf)  -- Visualize random batch.
_tr:visualizeBatch(_conf, _mconf, {1})  -- Explicitly define batch samples.
--]]

-- ************************ Profile the model for the paper ********************
if conf.profileFPROPTime > 0 then
  local res = 128  -- The 3D data is 64x64x64 (which isn't that interesting).
  print('==> Profiling FPROP for ' ..  conf.profileFPROPTime .. ' seconds' ..
        ' with grid res ' .. res)
  local nuchan, zdim
  if mconf.twoDim then
    nuchan = 2
    zdim = 1
  else
    nuchan = 3
    zdim = res
  end
  local batchGPU = {
     pDiv = torch.CudaTensor(1, 1, zdim, res, res):fill(0), 
     UDiv = torch.CudaTensor(1, nuchan, zdim, res, res):fill(0),
     geom = torch.CudaTensor(1, 1, zdim, res, res):fill(0),
  }
  model:evaluate()  -- Turn off training (so batch norm doesn't get messed up).
  local input = torch.getModelInput(batchGPU)
  model:forward(input)  -- Input once before we start profiling.
  cutorch.synchronize()  -- Make sure everything is allocated fully.
  local t0 = sys.clock()
  local t1 = t0
  local niters = 0
  while t1 - t0 < conf.profileFPROPTime do
    model:forward(input)
    cutorch.synchronize()  -- Flush the GPU buffer.
    t1 = sys.clock()
    niters = niters + 1
  end
  print('    FPROP Time: ' .. 1000 * ((t1 - t0) / niters) .. ' ms / sample')
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

  local logger = optim.Logger(conf.modelDirname .. '_log.txt')
  logger:setNames{'trLoss', 'trPLoss', 'trULoss', 'trDivLoss',
                  'trLongTermDivLoss', 'teLoss', 'tePLoss', 'teULoss',
                  'teDivLoss', 'teLongTermDivLoss'}

  if conf.profile then
    assert(ProFi ~= nil, 'cannot profile without "luarocks install ProFi"')
    ProFi:start()
  end

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

  if conf.profile then
    ProFi:stop()
    ProFi:writeReport(conf.modelDirname .. '_profileReport.txt')
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
local nSteps = 128
local teNormDiv = torch.calcStats(
    {data = te, conf = conf, mconf = mconf, model = model, nSteps = nSteps})
torch.save(conf.modelDirname .. '_teNormDiv.bin', teNormDiv)
local trNormDiv = torch.calcStats(
    {data = tr, conf = conf, mconf = mconf, model = model, nSteps = nSteps})
torch.save(conf.modelDirname .. '_trNormDiv.bin', trNormDiv)
if mattorch ~= nil then
  mattorch.save(conf.modelDirname .. '_teNormDiv.mat', 
                {meanDiv = teNormDiv:double()})
  mattorch.save(conf.modelDirname .. '_trNormDiv.mat',
                {meanDiv = trNormDiv:double()})
end

--[[
dofile('utils/plot_weight_histograms.lua')
local verbosePlot = false
torch.plotWeightHistograms(_conf, _mconf, _model, _tr, _criterion, verbosePlot)
--]]
