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

local function sleep(sec)  -- CAN BE FRACTIONAL!
  os.execute("sleep " .. tonumber(sec))
end

function torch.ls(dir)
  local files = {}
  --[[
  -- EDIT: We might be searching for a file within a directory, and the iterator
  -- is very slow for large directories.
  if paths.dirp(dir) == false then
    error('Directory ' .. dir .. ' does not exist')
  else
  --]]
  -- Collect the file and directory names
  -- This method returns a sorted list on unix
  for f in io.popen("ls " .. dir .. " 2> /dev/null"):lines() do
    table.insert(files, f)
  end
  --[[
  -- LFS doesn't return sorted arrays
  for file in lfs.dir(dir) do
    files[#files+1] = file
  end
  --]]
  -- end
  sleep(0.001)
  -- This is annoying, but if we don't add 1ms using throws interrupts.
  return files
end
