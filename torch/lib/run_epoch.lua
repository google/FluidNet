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

local cunn = require('cunn')
local sys = require('sys')
local tfluids = require('tfluids')
local mattorch = torch.loadPackageSafe('mattorch')

-- dofile('lib/load_package_safe.lua')

-- A single test AND train function (avoid code duplication). When training set
-- epochType == 'train' and when testing, epochType == 'test'.
function torch.runEpoch(input)
  -- Unpack the input table.
  local data = input.data
  local conf = input.conf
  local mconf = input.mconf
  local model = input.model
  local crit = input.criterion
  local parameters = input.parameters
  local gradParameters = input.gradParameters  -- OPTIONAL
  local optimMethod = input.optimMethod  -- OPTIONAL
  local epochType = input.epochType

  assert(epochType == 'train' or epochType == 'test')
  local training = epochType == 'train'

  if training then
    for i = 1, #conf.lrEpochMults do
      if mconf.epoch == conf.lrEpochMults[i].epoch then
        print('\n==> Appling learning rate multiplier ' ..
              conf.lrEpochMults[i].mult)
        mconf.optimState.learningRate =
            mconf.optimState.learningRate * conf.lrEpochMults[i].mult
      end
    end
  end

  torch.setDropoutTrain(model, training)

  local dataInds
  if training then
    -- When training, shuffle the indices.
    dataInds = torch.randperm(data:nsamples()):int()
  else
    -- When testing, go through the dataset in order.
    dataInds = torch.IntTensor(torch.range(1, data:nsamples()))
  end

  -- For fast debugging, we can reduce the epoch size.
  if conf.maxSamplesPerEpoch < math.huge then
    local nsamples = math.min(dataInds:size(1), conf.maxSamplesPerEpoch)
    dataInds = dataInds[{{1, nsamples}}]
  end

  -- Containers for the current batch.
  local batchGPU
  do
    local batchCPU = data:AllocateBatchMemory(conf.batchSize)
    batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')
  end

  -- Create a parallel data container to synchronously create batches (very
  -- important for the 2D model since diskIO becomes the bottleneck).
  local singleThreaded = false  -- Set to true for easier debugging.
  local parallel = torch.DataParallel(conf.numDataThreads, data, dataInds,
                                      conf.batchSize,
                                      torch.DataBinary.initThreadFunc,
                                      singleThreaded)

  if training then
    print('\n==> Training: (gpu ' .. tostring(conf.gpu) .. ')')
  else
    print('\n==> Testing: (gpu ' .. tostring(conf.gpu) .. ')')
  end
  print('  - criterion = '.. mconf.lossFunc)
  print("  - epoch # " .. mconf.epoch .. ' [bSize = ' .. conf.batchSize ..
        '] [learnRate = ' .. mconf.optimState.learningRate ..
        '] [optim = '.. mconf.optimizationMethod .. ']')
  io.flush()
  local nbatches = 0  -- Number processed
  local nbatchesTotal = math.ceil(dataInds:size(1) / conf.batchSize)
  local aveLoss = 0
  local avePLoss = 0
  local aveULoss = 0
  local aveDivLoss = 0
  local aveLongTermDivLoss = 0
  local lastBatchErr = nil
  local batchErr = {}  -- A list of batch indices and the crit err
  local time
  local processedSamples = {}  -- Hash set of samples we've processed.

  assert(not parallel:empty(), 'No batches to train / eval on!')

  repeat
    -- Get a processed batch.
    if math.fmod(nbatches, 10) == 0 then
      collectgarbage()
    end
    local progress_str = ''
    if lastBatchErr ~= nil then
      progress_str = string.format('err=%.4e', lastBatchErr)
    end
    torch.progress(nbatches * conf.batchSize, nbatchesTotal * conf.batchSize,
                   progress_str)

    -- Get a processed batch. NOTE: they may be out of order since we
    -- process them asynchronously. We only perturb during training.
    local batch = parallel:getBatch(conf.batchSize, conf.dataDir)

    -- Check we don't double count any samples. This is technically tested
    -- in test_data_parallel.lua, but do it again here just in case.
    local batchSet = batch.batchSet  -- Index set for the current batch.
    for i = 1, batchSet:size(1) do
      assert(processedSamples[batchSet[i]] == nil, 'Double counting sample!')
      processedSamples[batchSet[i]] = true
    end

    -- Sync the batch to the GPU.
    torch.syncBatchToGPU(batch.data, batchGPU)

    -- The first batch is not representative of processing time, don't include
    -- it (this is because lots of mallocs happen on the first batch).
    if time == nil then
      time = sys.clock()
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      local input = torch.getModelInput(batchGPU)
      local target = torch.getModelTarget(batchGPU)

      -- get new parameters
      if x ~= parameters then
        print('**** WARNING: Performing unnecessary copy in train.lua!')
        parameters:copy(x)
      end

      if training then
        -- reset gradients
        model:zeroGradParameters()  -- dE_dw
      end

      -- evaluate function for complete mini batch
      local output = model:forward(input)

      local err = crit:forward(output, target)
      lastBatchErr = err
      batchErr[#batchErr + 1] = {
          inds = batchSet,
          err = err,
      }

      local errLimit = 1e9
      if torch.isNan(err) or err > errLimit then
        print('')
        print('WARNING: criterion error (' .. err .. ') is NaN or > ' ..
              errLimit)
        error('criterion error is NaN or > 1e3.')
      end

      aveLoss = aveLoss + err
      if torch.type(crit) == 'nn.FluidCriterion' then
        avePLoss = avePLoss + crit.pLoss
        aveULoss = aveULoss + crit.uLoss
        aveDivLoss = aveDivLoss + crit.divLoss
      end

      if training then
        -- estimate df/dW.
        local df_do = crit:backward(output, target)
        model:backward(input, df_do)
      end

      -- Finally, we want to calculate the divergence of a future frame (and
      -- in some cases optimize this as well). This means we need to run the
      -- simulation in a loop to calculate the future frame.
      if mconf.longTermDivNumSteps ~= nil and mconf.longTermDivLambda > 0 then
        local baseDt = mconf.dt

        -- Pick a random timescale.
        if mconf.timeScaleSigma > 0 then
          -- note, randn() returns normal distribution with mean 0 and var 1.
          local scale = 1 + math.abs(torch.randn(1)[1] *
                                     mconf.timeScaleSigma)
          mconf.dt = baseDt * scale
        end

        torch.setDropoutTrain(model, false)

        local numFutureSteps = mconf.longTermDivNumSteps[1]
        local rand = math.random() -- Lua RNG. Uniform Dist [0,1]
        if rand > mconf.longTermDivProbability then
          numFutureSteps = mconf.longTermDivNumSteps[2]
        end

        for i = 1, numFutureSteps do
          local outputDiv = (i == numFutureSteps)
          tfluids.simulate(conf, mconf, batchGPU, model, outputDiv)
        end
        torch.setDropoutTrain(model, training)

        -- The simulate output is left on the GPU.
        local output = model:forward(input)

        -- Now calculate the divergence error for this future frame.
        assert(torch.type(crit) == 'nn.FluidCriterion')
        crit.pLambda = 0
        crit.uLambda = 0
        crit.divLambda = mconf.longTermDivLambda
        local errLongTermDiv = crit:forward(output, target)
        aveLongTermDivLoss = aveLongTermDivLoss + errLongTermDiv

        -- If we're including this in the objective function than also BPROP.
        err = err + errLongTermDiv
        aveLoss = aveLoss + errLongTermDiv
        local df_do = crit:backward(output, target)
        model:backward(input, df_do)

        -- Put the criterion lambdas back.
        -- TODO(tompson): This is a little ugly. Should we have a separate crit?
        crit.pLambda = mconf.lossPLambda
        crit.uLambda = mconf.lossULambda
        crit.divLambda = mconf.lossDivLambda

        -- Put the dt back.
        -- TODO(tompson): Again this is ugly, it might be safer to deep clone
        -- mconf.
        mconf.dt = baseDt
      end

      if gradParameters ~= nil then
        local curNorm = gradParameters:norm()

        -- Perform gradient clipping.
        if curNorm > mconf.gradNormThreshold then
          -- Rescale the L2 of the gradient to something reasonable.
          gradParameters:div(curNorm / mconf.gradNormThreshold)
        end
      end

      -- return f and df/dX
      return err, gradParameters
    end

    if training then
      -- optimize on current mini-batch using the above closure
      optimMethod(feval, parameters, mconf.optimState)
    else
      -- Just execute the callback to FPROP.
      feval(parameters)
    end

    nbatches = nbatches + 1
  until parallel:empty()

  -- Make sure we processed every single sample we meant to.
  assert(nbatches == nbatchesTotal)
  for i = 1, dataInds:size(1) do
    assert(processedSamples[dataInds[i]])
  end

  torch.progress(dataInds:size(1), dataInds:size(1))

  torch.setDropoutTrain(model, false)
  collectgarbage()

  -- time taken
  time = sys.clock() - time
  time = time / ((nbatches - 1) * conf.batchSize)
  print("  - Time to process 1 sample = " .. (time * 1000) .. 'ms')

  aveLoss = aveLoss / nbatches
  avePLoss = avePLoss / nbatches
  aveULoss = aveULoss / nbatches
  aveDivLoss = aveDivLoss / nbatches
  aveLongTermDivLoss = aveLongTermDivLoss / nbatches
  print("  - Current loss function value: " .. aveLoss)
  print("    - pLoss: " .. avePLoss ..
        ' (pLambda = ' .. mconf.lossPLambda .. ')')
  print("    - uLoss: " .. aveULoss ..
        ' (uLambda = ' .. mconf.lossULambda .. ')')
  print("    - divLoss: " .. aveDivLoss ..
        ' (divLambda = ' .. mconf.lossDivLambda .. ')')
  print("    - longTermDivLoss = " .. aveLongTermDivLoss .. ' (lambda = ' ..
        mconf.longTermDivLambda .. ', active = ' ..
        tostring(mconf.optimizeLongTermDiv) .. ')')
  if mconf.optimState.ms ~= nil then
    print("  - current mean squared value: " .. mconf.optimState.ms)
  end

  local ret = {
      loss = aveLoss,
      pLoss = avePLoss,
      uLoss = aveULoss,
      divLoss = aveDivLoss,
      longTermDivLoss = aveLongTermDivLoss,
      batchErr = batchErr,
  }
  return ret
end
