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

-- This is really just a 'catch all' function for a bunch of hacky data and
-- analysis (some of which was used for the paper).

local sys = require('sys')
local tfluids = require('tfluids')
local image = torch.loadPackageSafe('image')

-- Calculate divergence stats. 
function torch.calcStats(input)
  -- Unpack the input table.
  local data = input.data
  local conf = input.conf
  local mconf = input.mconf
  local model = input.model
  local nSteps = input.nSteps

  torch.setDropoutTrain(model, false)

  local batchCPU = data:AllocateBatchMemory(1)
  local batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')
  local divNet = tfluids.VelocityDivergence():cuda()

  -- Now lets go through the dataset and get the statistics of output
  -- divergence, etc.
  local batchCPU = data:AllocateBatchMemory(conf.batchSize)
  local batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')
  local dataInds = torch.randperm(data:nsamples()):int()
  if conf.maxSamplesPerEpoch < math.huge then
     local nsamples = math.min(dataInds:size(1), conf.maxSamplesPerEpoch)
     dataInds = dataInds[{{1, nsamples}}]
  end

  print('\n==> Calculating Stats: (gpu ' .. tostring(conf.gpu) .. ')')
  io.flush()
  local normDiv = torch.zeros(dataInds:size(1), nSteps):double()
  local nBatches = 0

  for t = 1, dataInds:size(1), conf.batchSize do
    if math.fmod(nBatches, 10) == 0 then
      collectgarbage()
    end
    torch.progress(t, dataInds:size(1))

    local imgList = {}  -- list of images in the current batch
    for i = t, math.min(math.min(t + conf.batchSize - 1, data:nsamples()),
                        dataInds:size(1)) do
      table.insert(imgList, dataInds[i])
    end

    -- TODO(tompson): Parallelize CreateBatch calls (as in run_epoch.lua), which
    -- is not easy because the current parallel code doesn't guarantee batch
    -- order (might be OK, but it would be nice to know what err each sample
    -- has).
    data:CreateBatch(batchCPU, torch.IntTensor(imgList), conf.batchSize,
                     conf.dataDir)

    -- We want to save the error distribution.
    torch.syncBatchToGPU(batchCPU, batchGPU)
    local input = torch.getModelInput(batchGPU)
    local target = torch.getModelTarget(batchGPU)
    local output = model:forward(input)
    local pPred, UPred = torch.parseModelOutput(output)
    local pTarget, UTarget, flags = torch.parseModelTarget(target)
    local pErr = pPred - pTarget
    local UErr = UPred - UTarget
     
    -- Now record divergence stability vs time.
    -- Restart the sim from the target frame.
    local p, U, flags, density = tfluids.getPUFlagsDensityReference(batchGPU)
    U:copy(batchGPU.UTarget)
    p:copy(batchGPU.pTarget)


    -- Try Zeroing the divergence using Jacobi (PCG would be too slow here).
    --[[
    mconf.trainTargetSource = 'jacobi'
    mconf.maxIter = 50
    tfluids.calcPUTargets(conf, mconf, batchGPU, model)
    --]]
    -- Just use the zero divergence manta frame. 
    p:copy(pTarget)
    U:copy(UTarget)

    -- Record the divergence of the start frame.
    div = divNet:forward({U, flags})
    local iout = t
    for i = 1, #imgList do
      normDiv[{iout, 1}] = div[i]:norm()
      iout = iout + 1
    end
    mconf.gravityScale = 0

    for j = 2, nSteps do
      local outputDiv = false
      tfluids.simulate(conf, mconf, batchGPU, model, outputDiv)
      local p, U, flags, density =
          tfluids.getPUFlagsDensityReference(batchGPU)
      div = divNet:forward({U, flags})
      local iout = t
      for i = 1, #imgList do
        normDiv[{iout, j}] = div[i]:norm()
        iout = iout + 1
      end
    end
    nBatches = nBatches + 1
  end
  torch.progress(dataInds:size(1), dataInds:size(1))  -- Finish the progress bar.

  return {normDiv = normDiv}
end
