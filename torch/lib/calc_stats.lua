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

local sys = require('sys')
local tfluids = require('tfluids')

-- Calculate divergence stats. 
function torch.calcStats(input)
  -- Unpack the input table.
  local data = input.data
  local conf = input.conf
  local mconf = input.mconf
  local model = input.model
  local nSteps = input.nSteps or 128

  torch.setDropoutTrain(model, false)
  local dataInds = torch.range(1, data:nsamples())

  local batchCPU, batchGPU = data:AllocateBatchMemory(conf, mconf)
  local matchManta = false
  local divNet = nn.VelocityDivergence(matchManta):cuda()

  print('\n==> Calculating Stats: (gpu ' .. tostring(conf.gpu) .. ')')
  io.flush()
  local normDiv = torch.FloatTensor(#dataInds, nSteps)
  local nBatches = 0
  for t = 1, #dataInds, conf.batchSize do
    if math.fmod(nBatches, 10) == 0 then
      collectgarbage()
    end
    torch.progress(t, #dataInds)

    local imgList = {}  -- list of images in the current batch
    for i = t, math.min(math.min(t + conf.batchSize - 1, data:nsamples()),
                        #dataInds) do
      table.insert(imgList, dataInds[i])
    end

    local perturbData = false
    data:CreateBatch(batchCPU, batchGPU, conf, mconf, imgList, perturbData)
    local p, U, geom, density = tfluids.getPUGeomDensityReference(batchGPU)
    -- Start with the DIVERGENCE FREE frame (not advected frame).
    U:copy(batchGPU.UTarget)
    p:copy(batchGPU.pTarget)

    -- Record the divergence of the start frame.
    local div = divNet:forward({U, geom})
    for i = 1, #imgList do
      normDiv[{imgList[i], 1}] = div[i]:norm()
    end

    for j = 2, nSteps do
      local outputDiv = false
      tfluids.simulate(conf, mconf, batchCPU, batchGPU, model, outputDiv)
      local p, U, geom, density =
          tfluids.getPUGeomDensityReference(batchGPU)
      div = divNet:forward({U, geom})
      for i = 1, #imgList do
        normDiv[{imgList[i], j}] = div[i]:norm()
      end
    end
    nBatches = nBatches + 1
  end
  torch.progress(#dataInds, #dataInds)  -- Finish the progress bar.

  return normDiv
end
