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

dofile('queue.lua')

local mytester = torch.Tester()
local tests = torch.TestSuite()

function tests.QueueFillSimple()
  -- Fill the Queue with a bunch of values and make sure we get them back, also
  -- check the valid empty states.
  local queue = torch.Queue()
  mytester:assert(queue:empty())
  mytester:asserteq(queue:numElements(), 0)

  local ok, val = queue:pop()
  mytester:assert(not ok)
  mytester:assert(val == nil)

  local numVals = 100
  local vals = torch.rand(numVals)

  for i = 1, numVals do
    queue:push(vals[i])
    mytester:assert(not queue:empty())
    mytester:asserteq(queue:numElements(), i) 
  end

  for i = 1, numVals do
    mytester:assert(not queue:empty())
    local ok, val = queue:pop()
    mytester:assert(ok)
    mytester:asserteq(vals[i], val)
    mytester:asserteq(queue:numElements(), numVals - i)
  end

  mytester:assert(queue:empty())
  mytester:asserteq(queue:numElements(), 0)

  ok, val = queue:pop()
  mytester:assert(not ok)
  mytester:assert(val == nil)
end

function tests.QueueInterleavedPushPop()
  local queue = torch.Queue()
  local numVals = 100
  local vals = torch.rand(numVals * 2)
  local iout = 1
  local ipop = 1
  for i = 1, numVals do
    mytester:asserteq(queue:numElements(), i - 1)
    -- Push twice.
    queue:push(vals[iout])
    queue:push(vals[iout + 1])
    iout = iout + 2
    mytester:asserteq(queue:numElements(), i + 1)

    -- Pop once
    local ok, val = queue:pop()
    mytester:assert(ok)
    mytester:asserteq(vals[ipop], val)
    ipop = ipop + 1
    mytester:asserteq(queue:numElements(), i)
  end

  -- We should now have numVals in the queue.
  -- Now empty the remaining values.
  for i = 1, numVals do
    local ok, val = queue:pop()
    mytester:assert(ok)
    mytester:asserteq(vals[ipop], val)
    ipop = ipop + 1
  end

  mytester:assert(queue:empty())
end

function tests.QueueStoreNil()
  local queue = torch.Queue()
  local ok, val = queue:pop()
  mytester:assert(not ok)
  mytester:assert(val == nil)

  queue:push(nil)
  mytester:assert(not queue:empty())
  
  ok, val = queue:pop()
  mytester:assert(ok)
  mytester:assert(val == nil)
  mytester:assert(queue:empty())

  ok, val = queue:pop()
  mytester:assert(not ok)
  mytester:assert(val == nil)
end

-- Now run the test above
mytester:add(tests)
mytester:run()
