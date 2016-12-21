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

-- This is a pretty messy top level include script. Note that not all
-- libraries are needed for every simulation run (however a warning will be
-- printed anyway if they cannot be loaded).

-- When using our internal interpreter, we need to redirect torch.xxx file
-- methods.
local result, file = pcall(require, 'learning.lua.file')
if result and file ~= nil then
  print('WARNING: redirecting file functions (open, save, filep).')
  io.open = file.open
  torch.load = function(filename)
    assert(file.Exists(filename))
    local object = file.LoadObject(filename)
    return object
  end

  torch.save = function(filename, data)
    file.SaveObject(filename, data)
  end

  paths.filep = function(filename)
    return file.Exists(filename)
  end
end

-- Define an agnostic mkdir function.
torch.mkdir = function(dirname)
  if result and file ~= nil then
    assert(file.MakeDir(dirname))
  else
    os.execute('mkdir -p "' .. dirname .. '"')
  end
end
