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

-- Kris's voxel utils.

local torch = require('torch')
local tfluids = require('tfluids')

function tfluids.calculateBoundingBox(voxels)
  assert(voxels:dim() == 3)
  local dims = voxels:size()
  local max = {1, 1, 1}
  local min = dims

  local count = 0
  for z = 1, dims[3] do
    for y = 1, dims[2] do
      for x = 1, dims[1] do
        if voxels[{x, y, z}] ~= 0 then
          count = count + 1
          if x < min[1] then min[1] = x end 
          if x > max[1] then max[1] = x end 

          if y < min[2] then min[2] = y end 
          if y > max[2] then max[2] = y end 

          if z < min[3] then min[3] = z end 
          if z > max[3] then max[3] = z end 
        end
      end
    end
  end
  assert(count > 0) -- Make sure the volume wasn't empty.
  return {min=min, max=max}
end

function tfluids.intersectionTestAABBtoAABB(a, b)
  assert(a.min ~= nil and b.min ~= nil)
  assert(a.max ~= nil and b.max ~= nil)
  if (a.min[1] > b.max[1]  or  a.max[1] < b.min[1]) then return false end
  if (a.min[2] > b.max[2]  or  a.max[2] < b.min[2]) then return false end
  if (a.min[3] > b.max[3]  or  a.max[3] < b.min[3]) then return false end
  return true
end

--  Does the other AABB lie completely inside the src AABB?
function tfluids.doesAABBEncompassOther(src, other)
  assert(src.min ~= nil and other.min ~= nil)
  assert(src.max ~= nil and other.max ~= nil)
  if (
      src.min[1] <= other.min[1] and src.max[1] >= other.max[1] and
      src.min[2] <= other.min[2] and src.max[2] >= other.max[2] and
      src.min[3] <= other.min[3] and src.max[3] >= other.max[3] 
     ) then 
    return true
  end
  return false
end

function tfluids.isVoxelInBounds(grid, point)
  assert(#point == 3)
  assert(#grid:size() == 3)
  assert(point[1] >= 1 and point[2] >= 1 and point[3] >= 1)

  local dims = grid:size()
  if (point[1] <= dims[1] and point[2] <= dims[2] and point[3] <= dims[3]) then
    return true
  end
  return false
end

function tfluids.expandAABInfo(a)
  assert(a.min ~= nil and a.max ~= nil)

  local dims = {a.max[1] - a.min[1], a.max[2] - a.min[2], a.max[3] - a.min[3]} 
  assert(dims[1] > 0 and dims[2] > 0 and dims[3] > 0)

  local halfdims = {dims[1] * 0.5, dims[2] * 0.5, dims[3] * 0.5}
  local center = {a.min[1] + halfdims[1], a.min[2] + halfdims[2], a.min[3] +
      halfdims[3]}
  assert(
      center[1] == (a.max[1] - a.min[1]) * 0.5 and
      center[2] == (a.max[2] - a.min[2]) * 0.5 and
      center[3] == (a.max[3] - a.min[3]) * 0.5 
      )
  a.dims = dims
  a.halfdims = halfdims
  a.center = center
  return a
end

function tfluids.translateVoxels(voxels, shift)
  local dims = voxels:size()
  -- Now shift the voxels.  This can be done by simple index offsets
  -- But be carefull to bounds check the lookup into the voxel source
 
  -- TODO(kris): This is slow. There is probably a faster way using slicing. 
  -- At the very least I can make these arrays contiguous and use indexing
  -- into the .data.              
  -- However this does not need to be fast as it is only done once at the 
  -- beginning of a simulation.
  local tmp = torch.FloatTensor(dims[1], dims[2], dims[3]):fill(0)
  for i = 1, dims[1] do
    local newi = i + shift[1]
    for j = 1, dims[2] do
      local newj = j + shift[2]
      for k = 1, dims[3] do
        local newk = k + shift[3]
        if newi <= dims[1] and newj <= dims[2] and newk <= dims[3] and
             newi > 0 and newj > 0 and newk > 0 then
          tmp[{newi, newj, newk}] = voxels[{i, j, k}]
        end
      end 
    end 
  end 
  voxels:copy(tmp)
end

function tfluids.moveVoxelCentroidToCenter(voxels)
  local dims = voxels:size()

  -- Calculate Centroid.
  local voxelCount = 0
  local values = {0, 0, 0} -- Has to be per plane.  1 = x/y, 2 = x/z, 3 = z/y.
  for i = 1, dims[1] do
    for j = 1, dims[2] do
      for k = 1, dims[3] do
        if voxels[{i, j, k}] > 0 then
          values[1] = values[1] + i
          values[2] = values[2] + j
          values[3] = values[3] + k
          voxelCount = voxelCount + 1
        end
      end
    end
  end
  if (voxelCount > 0) then
    local centroid = {values[1] / voxelCount, values[2] / voxelCount, values[3]
        / voxelCount}
    local center = {dims[1] * 0.5, dims[2] * 0.5, dims[3] * 0.5}
    local shift = {math.floor(center[1] - centroid[1]), math.floor(center[2] - 
        centroid[2]), math.floor(center[3] - centroid[3])}

    tfluids.translateVoxels(voxels, shift)
  end
end


function tfluids.moveVoxelBBoxToCenter(voxels)
  local bbox = tfluids.calculateBoundingBox(voxels)
  local dims = voxels:size()
  local sizes = {bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2],
    bbox.max[3] - bbox.min[3]}
  local extra = {dims[1] - sizes[1], dims[2] - sizes[2], dims[3] - sizes[3]}
  local shift = {extra[1] * 0.5, extra[2] * 0.5, extra[3] * 0.5}

  tfluids.translateVoxels(voxels, shift)
end

function tfluids.expandVoxelsToDims(width, height, depth, target)
  assert(#target:size() == 3)
  local dims = target:size()
  assert(dims[1] <= depth and dims[2] <= height and dims[3] <= width)
  local tmp = torch.FloatTensor(depth, height, width):fill(0)
  tmp[{{1, dims[1]}, {1, dims[2]}, {1, dims[3]}}] = target
  return tmp
end

function tfluids.blitIntoTarget(src, target, offset)
  assert(#src:size() == 3 and #target:size() == 3)
  assert(#offset == 3)

  local srcDims = src:size()
  local targetDims = target:size()

  assert(sdims[1] <= dims[1] and sdims[2] <= dims[2] and sdims[3] <= sdims[3])
  local startIdx = {int(offset[1]), int(offset[2]), int(offset[3])}
  local endIdx = {int(offset[1] + sdims[1]), int(offset[2] + sdims[2]),
      int(offset[3] + sdims[3])}
  assert(offset[1] >= 0 and offset[2] >= 0 and offset[3] >= 0)
  assert(sdims[1] >= 0 and sdims[2] >= 0 and sdims[3] >= 0)
  assert(endIdx[1] <= dims[1] and endIdx[2] <= dims[2] and endIdx[3] <= dims[3])

  target[{{startIdx[1],endIdx[1]}, {startIdx[2],endidx[2]},
      {startIdx[3],endIds[3]}}] = src
end

-- Used to be named: tfluids.rotateVoxels90
function tfluids.flipDiagonal(voxels, axis)
  local dims = voxels:size()
  assert(#dims == 3)
  assert(axis >=0 and axis <=2)
  if axis == 0 then
    assert(dims[2] == dims[3])
  elseif axis == 1 then
    assert(dims[1] == dims[3])
  else
    assert(dims[1] == dims[2])
  end

  -- I have to use a copy.  I tried in place swaps and it didn't work
  local tmp = torch.FloatTensor(dims[1], dims[2], dims[3]):fill(0)
  for i = 1, dims[1] do
    for j = 1, dims[2] do
      for k = 1, dims[3] do
          local ii = i
          local jj = j
          local kk = k
          if axis == 0 then
            ii = i
            jj = k
            kk = j
          elseif axis == 1 then
            ii = k
            jj = j
            kk = i
          else
            ii = j
            jj = i
            kk = k
          end
          tmp[{i, j, k}] = voxels[{ii, jj, kk}]
          tmp[{ii, jj, kk}] = voxels[{i, j, k}]
      end
    end
  end
  voxels:copy(tmp)
end

function tfluids.flipVoxels(voxels, axis)
  local dims = voxels:size()
  assert(#dims == 3)
  assert(axis >=0 and axis <=2)

  local halfi = dims[1] * 0.5
  local halfj = dims[2] * 0.5
  local halfk = dims[3] * 0.5
  local maxi, maxj, maxk
  maxi = dims[1]
  maxj = dims[2]
  maxk = dims[3]
  if axis == 0 then
    maxi = halfi
  elseif axis == 1 then
    maxj = halfj
  else
    maxk = halfk
  end
  -- I have to use a copy.  I tried in place swaps and it didn't work
  local tmp = torch.FloatTensor(dims[1], dims[2], dims[3]):fill(0)
  for i = 1, maxi do
    for j = 1, maxj do
      for k = 1, maxk do
          local ii = i
          local jj = j
          local kk = k
          if axis == 0 then
            ii = dims[1] - i
            jj = j
            kk = k
          elseif axis == 1 then
            ii = i
            jj = dims[2] - j
            kk = k
          else
            ii = i
            jj = j
            kk = dims[3] - k
          end
          tmp[{i, j, k}] = voxels[{ii, jj, kk}]
          tmp[{ii, jj, kk}] = voxels[{i, j, k}]
      end
    end
  end
  voxels:copy(tmp)

end
