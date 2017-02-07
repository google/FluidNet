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

function torch.loadMantaFile(fn)
  assert(paths.filep(fn), "couldn't find ".. fn)
  local file = torch.DiskFile(fn, 'r')
  file:binary()
  local transpose = file:readInt()  -- Legacy. Never used.
  local nx = file:readInt()
  local ny = file:readInt()
  local nz = file:readInt()
  local is3D = file:readInt()
  is3D = is3D == 1
  local numel = nx * ny * nz
  local Ux = torch.FloatTensor(file:readFloat(numel))
  local Uy = torch.FloatTensor(file:readFloat(numel))
  local Uz
  if is3D then
    Uz = torch.FloatTensor(file:readFloat(numel))
  end
  local p = torch.FloatTensor(file:readFloat(numel))
  -- Note: flags are stored as integers but well treat it as floats to make it
  -- easier to sent to CUDA.
  local flags = torch.IntTensor(file:readInt(numel)):float()
  local density = torch.FloatTensor(file:readFloat(numel))

  -- Note: we ALWAYS deal with 5D tensor to make things easy.
  -- i.e. all tensors are always nbatch x nchan x nz x ny x nx.
  -- Always including the batch and (potentially) unary scalar dimensions
  -- makes our lives easier.
  Ux:resize(1, 1, nz, ny, nx)
  Uy:resize(1, 1, nz, ny, nx)
  if is3D then
    Uz:resize(1, 1, nz, ny, nx)
  end
  p:resize(1, 1, nz, ny, nx)
  flags:resize(1, 1, nz, ny, nx)
  density:resize(1, 1, nz, ny, nx)

  local U
  if is3D then
    U = torch.cat({Ux, Uy, Uz}, 2):contiguous()
  else
    U = torch.cat({Ux, Uy}, 2):contiguous()
  end

  -- Ignore the border pixels.
  return p, U, flags, density, is3D
end

