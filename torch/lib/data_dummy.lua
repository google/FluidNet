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

-- Create a dummy data handling class to test the DataParallel container.
local DataDummy, parent = torch.class('torch.DataDummy')

function DataDummy:__init(numSamples, workTime)
  self.numSamples = numSamples
  self.workTime = workTime
  -- Each "sample" will just be a random number. So we'll map the sample
  -- index to a random value to make sure the batches are properly formatted.
  self.sampleData = torch.FloatTensor(numSamples):random()
end

function DataDummy:AllocateBatchMemory(batchSize, ...)
  local batchCPU = {
      sampleData = torch.FloatTensor(batchSize):fill(-1),
  } 
  return batchCPU
end

function DataDummy:CreateBatch(batchCPU, sampleSet, ...)
  local batchSize = sampleSet:size(1)  -- Not necessarily the same as container
  -- Make sure the container is not too small.
  assert(batchSize <= batchCPU.sampleData:size(1))
  for i = 1, batchSize do
    batchCPU.sampleData[i] = self.sampleData[sampleSet[i]]
  end
  -- Simulate work being done in the thread.
  os.execute("sleep " .. tonumber(self.workTime))
end

