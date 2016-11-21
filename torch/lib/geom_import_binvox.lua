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

local torch = require('torch')
local tfluids = require('tfluids')
dofile("lib/voxel_utils.lua")

-- Function found on http://lua-users.org/wiki/SplitJoin
function split(str, pat)
  local t = {}  -- NOTE: use {n = 0} in Lua-5.0
  local fpat = "(.-)" .. pat
  local last_end = 1
  local s, e, cap = str:find(fpat, 1)
  while s do
    if s ~= 1 or cap ~= "" then
      table.insert(t,cap)
    end
    last_end = e + 1
        s, e, cap = str:find(fpat, last_end)
  end
  if last_end <= #str then
    cap = str:sub(last_end)
    table.insert(t, cap)
  end
  return t
end

function tfluids.loadVoxelFileHeader(file)
  local binvoxName = file:readString("*l")
  local dimTable = split(file:readString("*l"), " +")
  local translateTable = split(file:readString("*l"), " +")
  local scaleTable = split(file:readString("*l"), " +")
  local dataString = file:readString("*l")

  local dims = {dimTable[2], dimTable[3], dimTable[4]}
  local translation = {translateTable[2], translateTable[3], translateTable[4]}
  local scale = scaleTable[2]
  return dims, translation, scale
end

function tfluids.loadVoxelData(filename)
  local file = torch.DiskFile(filename, 'r')
  if file == nil then
    print("Could not open " .. filename)
    return
  else
    print("Reading " .. filename)
  end
  file:pedantic()
  file:binary()

  local dims, translation, scale = tfluids.loadVoxelFileHeader(file)

  local dataStartPosition = file:position()
  file:seekEnd()
  local endPosition = file:position()
  local dataBytesCount = endPosition - dataStartPosition
  file:seek(dataStartPosition)
  local voxelCount = dims[1] * dims[2] * dims[3]

  local data1D = torch.ByteTensor(voxelCount)
  data1D:fill(0)

  -- rawdata data consists of pairs of bytes. The first byte of each pair is
  -- the value byte and is either 0 or 1 (1 signifies the presence of a voxel).
  -- The second byte is the count byte and specifies how many times the
  -- preceding voxel value should be repeated (so obviously the minimum count
  -- is 1, and the maximum is 255)

  -- This is a port of their C++ parser: 
  -- http://www.patrickmin.com/binvox/read_binvox.html
  local index = 1
  local endIndex = 1
  local numVoxelsRead = 0
  while ((endIndex < voxelCount) and (file:position() < endPosition)) do
    local value = file:readByte(1)[1]
    local count = file:readByte(1)[1]

    if (file:position() < endPosition) then
      endIndex = index + count
      if (endIndex > voxelCount) then 
        print("Voxelreading somehow messed up indexing... ending read early!!")
        return 0
      end
      for i = index, endIndex do
        data1D[i] = value
      end

      if value > 0 then
        numVoxelsRead = numVoxelsRead + count
      end
      index = endIndex
    end

  end
  print("read " .. numVoxelsRead .. " voxels out of " .. voxelCount)

  local data = data1D:view(torch.LongStorage{dims[1], dims[2],
      dims[3]}):permute(1, 3, 2)

  local count = 0
  print("iterating through " .. voxelCount .. " voxels")
  for z = 1, dims[3] do
    for y = 1, dims[2] do
      for x = 1, dims[1] do
        if data[{ x, y, z}] > 0 then
          count = count + 1
        end
      end
    end
  end
  print("Found " .. count .. " voxels, expected: " .. numVoxelsRead)
  file:close()

  tfluids.calculateBoundingBox(data)

  return {dims=dims, translation=translation, scale=scale, data=data:float()}
end
