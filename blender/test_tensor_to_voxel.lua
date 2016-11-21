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

require('torch')

-- Just an example to write out a voxel grid so blender can read it.
local dim = 64
local frames = 8
local arr = torch.FloatTensor(frames, dim, dim, dim)
arr:zero()
arr[{{}, {2, 62}, {24, 40}, {24, 40}}] = 0.9

-- From docs on file.lua. Serialization methods.
local file = torch.DiskFile('lua_test.vbox','w')
file:binary()
file:writeInt(dim)
file:writeInt(dim)
file:writeInt(dim)
file:writeInt(frames)
file:writeFloat(arr:permute(1, 4, 3, 2):contiguous():storage())
file:close()
