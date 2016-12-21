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

-- This class is a simple push and pop queue (single-ended FIFO structure).
-- It is NOT thread-safe.

-- Note: this implementation guards against pointer overflow, to cover the case
-- if you do not exhaust the queue (we run two pointers, one for the head and
-- tail of the queue). We artificially set the maximum head value as 2^53
-- since lua uses doubles to represent numbers. This should be enough for all
-- but the most extreme use cases.
local Queue, parent = torch.class('torch.Queue')

local MAX_INTEGER = 9007199254740992  -- 2^53

function Queue:__init()
  self._queue = {}
  self._head = 1
  self._tail = 0
end

function Queue:push(value)
  self._tail = self._tail + 1
  self._queue[self._tail] = value 
  assert(self._tail <= MAX_INTEGER, 'Queue overflow error!')
end

-- @return - 2 values. boolean and a value. The boolean indicates whether there
-- was an element to pop (false if queue is empty). The value is the popped
-- element (nil if the queue is empty).
-- Note: we cannot just have a single return with nil for empty, since we might
-- want to store nil values in the queue.
function Queue:pop()
  if self:empty() then
    return false, nil
  end
  local value = self._queue[self._head]
  self._queue[self._head] = nil  -- so it is garbage collected properly.
  self._head = self._head + 1

  if self:empty() then
    -- Allow a queue pointer "reset" to reduce the chance of overflow.
    -- Obviously if the user never empties the queue, this does not help.
    self._head = 1
    self._tail = 0
  end

  return true, value
end

function Queue:empty()
  return self:numElements() <= 0
end

function Queue:numElements()
  return self._tail - self._head + 1
end
