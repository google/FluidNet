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

require 'nn'
local sys = require('sys')

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(16)

dofile('queue.lua')
dofile('data_parallel.lua')
dofile('data_dummy.lua')

-- Create an instance of the test framework
local mytester = torch.Tester()
local jac = nn.Jacobian
local tests = torch.TestSuite()

local function initThreadFunc()
  -- Threads must redefine all classes they use!
  require('torch')
  require('nn')
  dofile('data_parallel.lua')
  dofile('data_dummy.lua')
end

local function checkBatchData(batches, dataProvider, batchSize, numSamples)
  -- Make sure each batch is properly formatted.
  for _, batch in pairs(batches) do
    -- Each batch sample should have a batchSet (the set of indices).
    mytester:assert(batch.batchSet ~= nil)
    mytester:assert(torch.isTensor(batch.batchSet))
    mytester:assert(batch.batchSet:size(1) <= batchSize)  -- Could be partial.
    -- Each batch should have some data.
    mytester:assert(batch.data ~= nil)
    mytester:assert(torch.isTensor(batch.data.sampleData))
    mytester:assert(batch.data.sampleData:size(1) == batchSize)
  end

  -- Make sure each sample index was covered exactly once.
  local hashSet = {}
  for _, batch in pairs(batches) do
    local batchSet = batch.batchSet
    for i = 1, batchSet:size(1) do
      local curSampleInd = batchSet[i]
      mytester:assert(hashSet[curSampleInd] == nil)
      hashSet[curSampleInd] = true
    end
  end
  mytester:asserteq(#hashSet, numSamples)

  -- Make sure each sample value is what we expect.
  for _, batch in pairs(batches) do
    local batchSet = batch.batchSet
    for i = 1, batchSet:size(1) do
      local curSampleInd = batchSet[i]
      local curSampleValue = batch.data.sampleData[i]
      mytester:asserteq(dataProvider.sampleData[curSampleInd], curSampleValue)
    end
  end
end

function tests.ParallelCreateBatch()
  local numThreads = 16
  local batchSize = 7
  local numBatches = 100
  local workTime = 0.01  -- Seconds.
  local numSamples = batchSize * (numBatches - 1) + 1  -- Create 1 partial batch

  local dataProvider = torch.DataDummy(numSamples, workTime)
  local sampleSet = torch.randperm(numSamples)  -- All samples in the epoch.
  local parallel = torch.DataParallel(numThreads, dataProvider, sampleSet,
                                      batchSize, initThreadFunc)

  mytester:assert(not parallel:empty())

  local batches = {}  -- We'll save all the batches.
  repeat
    local batch = parallel:getBatch()
    mytester:assert(batch ~= nil)
    batches[#batches + 1] = batch
    mytester:assertle(#batches, numBatches)  -- Make sure we don't get too many
  until parallel:empty()

  mytester:assert(parallel:empty())  -- redundant, but do it anyway.

  checkBatchData(batches, dataProvider, batchSize, numSamples)
end

function tests.SingleThreadedCreateBatch()
  local numThreads = nil
  local batchSize = 7
  local numBatches = 100
  local workTime = 0.01  -- Seconds.
  local numSamples = batchSize * (numBatches - 1) + 1  -- Create 1 partial batch
  local singleThreaded = true

  local dataProvider = torch.DataDummy(numSamples, workTime)
  local sampleSet = torch.randperm(numSamples)  -- All samples in the epoch.
  local parallel = torch.DataParallel(numThreads, dataProvider, sampleSet,
                                      batchSize, initThreadFunc, singleThreaded)

  mytester:assert(not parallel:empty())

  local batches = {}  -- We'll save all the batches.
  repeat
    local batch = parallel:getBatch()
    mytester:assert(batch ~= nil)
    batches[#batches + 1] = batch
    mytester:assertle(#batches, numBatches)  -- Make sure we don't get too many
  until parallel:empty()

  mytester:assert(parallel:empty())  -- redundant, but do it anyway.

  checkBatchData(batches, dataProvider, batchSize, numSamples)
end

function tests.profileDataParallel()
  local batchSize = 7
  local numBatches = 50
  local workTime = 0.1  -- Seconds.
  local numSamples = batchSize * (numBatches - 1) + 1  -- Create 1 partial batch

  local dataProvider = torch.DataDummy(numSamples, workTime)
  local sampleSet = torch.randperm(numSamples)  -- All samples in the epoch.

  local timing = {}
  local numPowers2 = 5
  for i = 0, numPowers2 - 1 do
    local numThreads = math.pow(2, i)
    local parallel = torch.DataParallel(numThreads, dataProvider, sampleSet,
                                        batchSize, initThreadFunc)
    sys.tic()
    -- Exhaust the queue
    local numProcessedBatches = 0
    repeat
      parallel:getBatch()
      mytester:assertle(numProcessedBatches, numBatches)
      numProcessedBatches = numProcessedBatches + 1
    until parallel:empty()
    local time = sys.toc()
    timing[numThreads .. ' threads'] = time
  end

  print('Timing numers (batchSize = ' .. batchSize .. ', numBatches = ' ..
        numBatches .. ', workTime = ' .. workTime .. '):')
  print(timing)
end

-- Now run the test above
mytester:add(tests)
mytester:run()
