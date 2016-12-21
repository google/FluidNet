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

-- For public torch distros, the threads package is required.
-- >> luarocks install threads

local res, threads = pcall(require, 'third_party.lua.threads')
if res == false then
  print('INFO: third_party.lua.threads library not found, using threads')
  threads = require('threads')
end

-- This is a utility class that encapsulates parallelizing createBatch calls on
-- a data handler object. i.e. it is useful to hide the processing latency of
-- data fetching and pre-processing.
--
-- The dataProvider object is expected to have 2 methods implemented:
--
-- 1. batchCPU = dataProvider.AllocateBatchMemory(batchSize, ...)
--
-- which, returns a table of batch data (of unspecified format). '...' arg will
-- be variable number of args propagated down from getBatch() (so you can pass
-- in whatever you need).
--
-- 2. dataProvider.CreateBatch(batchCPU, sampleSet, ...)
--
-- which creates a batch in-place in the batchCPU container of samples with
-- indices in sampleSet. Note: you should NOT use any shared memory in
-- CreateBatch (because threads will clash).
--
-- See test_data_parallel.lua for a usage example.
-- tl;dr A DataParallel object is created once per epoch to iterate over
-- a set of sample ids in batchSize chunks.
--
-- Note: the DataParallel API is NOT thread safe. That is we expect only one
-- main thread calling class methods on the DataParallel object.

local DataParallel, parent = torch.class('torch.DataParallel')

-- Non-blocking constructor. Spawns worker threads that start populating
-- data results.
-- @param numThreads - number of workther threads.
-- @param dataProvider - Instance of the data provider class described above.
-- @param sampleSet - 1D Tensor of sample indices.
-- @param batchSize - size of each batch.
-- @param initThreadFunc - Lambda function to initialize each thread. NOTE:
-- threads start WITHOUT any definitions or classes defined. Therefore, you
-- will need to redefine anything used by the thread (i.e. redefine the
-- dataProvider class).
-- @param singleThreaded - (OPTIONAL) Used for debugging. Serializes all
-- createBatch calls into the main thread.
function DataParallel:__init(numThreads, dataProvider, sampleSet, batchSize,
                             initThreadFunc, singleThreaded)
  if singleThreaded == nil then
    singleThreaded = false
  end
  if not singleThreaded then
    assert(numThreads > 0)
  end
  -- Start the threads.
  assert(torch.isTensor(sampleSet) and sampleSet:dim() == 1)

  self._sampleSet = sampleSet
  self._batchSize = batchSize
  self._nextSampleIndex = 1
  self._singleThreaded = singleThreaded
  self._dataProvider = dataProvider
  self._batchSize = batchSize

  -- Create the thread queue.
  if not self._singleThreaded then
    self._curBatch = {}
    self._pool = threads.Threads(
        numThreads,
        -- Note: According to the docs, you must init all function definitions
        -- first before storing any types in the thread local storage.
        function(tid)
          initThreadFunc()
        end,
        function(tid)
          threadDP = dataProvider  -- Thread local variable.
          threadID = tid
          threadBS = batchSize
        end
    )
  end
end

function DataParallel:_getNextBatchSet()
  local batchSize = self._batchSize  -- Can't upvalue self.xxx values.
  local batchEndIndex = math.min(self._nextSampleIndex + batchSize - 1,
                                 self._sampleSet:size(1))
  local batchSet = self._sampleSet[{{self._nextSampleIndex, batchEndIndex}}]
  self._nextSampleIndex = self._nextSampleIndex + self._batchSize
  return batchSet
end

-- fillQueue will fill up the queue as much as possible.
function DataParallel:_fillQueue(...)
  if self._singleThreaded then
    return
  end

  local args = {...}

  -- Full up the queue as much as we can.
  while self._nextSampleIndex <= self._sampleSet:size(1) and
      self._pool:acceptsjob() do

    -- Add a batch of samples.
    local batchSet = self:_getNextBatchSet()

    -- Add a job for the current batchSet.
    local function threadFunc()
      -- Allocate the batch memory.
      local batchCPU = threadDP:AllocateBatchMemory(threadBS, unpack(args))

      -- Create the batch.
      threadDP:CreateBatch(batchCPU, batchSet, unpack(args))

      -- Return the batch data.
      return {data = batchCPU, batchSet = batchSet}
    end
    self._pool:addjob(
      threadFunc,
      function(jobres)
        self._curBatch = jobres
      end
    )
  end
end

-- returns true if there are still batches in the queue.
function DataParallel:empty()
  if self._nextSampleIndex <= self._sampleSet:size(1) then
    return false  -- Still batches to be placed on the queue.
  end

  -- Single threaded mode is not asynchronous. Nothing should be pending, if
  -- we're beyond the sample set end then we're done.
  if self._singleThreaded then
    return true
  end

  -- Otherwise, see if the queue is empty and has been fully depleted.
  return not self._pool:hasjob()
end

-- Get a processed batch - blocking until a batch is ready from a thread.
-- Note: if there are no more samples to process, the method will return nil,
-- otherwise will return the batch sample.
--
-- @param '...' - args for createBatch and allocateBatchMemory functions.
-- @return batch data if samples are left in the epoch, else nil. All data
-- ownership is transferred to the caller.
function DataParallel:getBatch(...)
  local args = {...}

  if self._singleThreaded then
    local batchSet = self:_getNextBatchSet()
    local batchCPU =
        self._dataProvider:AllocateBatchMemory(self._batchSize, unpack(args))
    self._dataProvider:CreateBatch(batchCPU, batchSet, unpack(args))
    return {data = batchCPU, batchSet = batchSet}
  end

  -- Add samples to the queue.
  self:_fillQueue(unpack(args))

  -- If the queue is still empty, then return nil.
  if self:empty() then
    return nil
  end

  self._pool:dojob()  -- Blocking until a batch is created.
  -- Result is now in self._curBatch

  -- Try adding items to the queue again.
  self:_fillQueue(unpack(args))

  -- Return the processed data.
  return self._curBatch
end

