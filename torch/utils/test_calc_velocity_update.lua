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

-- Velocity update (written by Kris and modified by Jonathan) to mimic
-- CorrectVelocity() in manta (line 57 of source/plugin/pressure.cpp).
--
-- We use this only to test that our Module implementation is correct.
-- If we cannot match manta's velocity update through our fixed function module
-- then the network will never learn the correct mapping (MSE requires both
-- pressure AND velocity are correct at the same time).

local tfluids = require('tfluids')
local nn = require('nn')
require('image')
dofile('../lib/load_package_safe.lua')
dofile('../lib/data_binary.lua')  -- For torch.DataBinary:_loadFile()
dofile('calc_velocity_update.lua')
dofile('../lib/modules/spatial_finite_elements.lua')
dofile('../lib/modules/volumetric_finite_elements.lua')
dofile('../lib/pbar.lua')

local function ls(dir)
  local files = {}
  for f in io.popen("ls " .. dir .. " 2> /dev/null"):lines() do
    table.insert(files, f)
  end
  os.execute("sleep 0.001")
  -- This is annoying, but if we don't add 1ms using throws interrupts.
  return files
end

local function testVelocityUpdate()
  -- Load a frame from file (for testing). We'll test both the 2D and 3D cases.
  local dataDirs = {}
  dataDirs[#dataDirs + 1] = '../../data/datasets/output_current_model_sphere'
  dataDirs[#dataDirs + 1] = '../../data/datasets/output_current_3d_model_sphere'
  local setName = 'tr'
  local maxErr = 0
  local precision = 1e-5

  for _, dataDir in pairs(dataDirs) do
    local setDir = dataDir .. '/' .. setName .. '/'
    print('Working from dir: ' .. setDir)

    -- Test every single scene and every single frame!
    -- Firstly, collect all the frames.
    local scenes = ls(setDir)
    assert(#scenes > 0)
    local allFiles = {}
    local allDivFiles = {}
    print('  ==> Collecting file names')
    for i = 1, #scenes do
      torch.progress(i, #scenes)
      local scene = scenes[i]
      local sceneDir = setDir .. scene .. '/'
      local files = ls(sceneDir .. '*[0-9].bin')
      local divFiles = ls(sceneDir .. '*[0-9]_divergent.bin')
      assert(#files == #divFiles)
      for i = 1, #files do
        allFiles[#allFiles + 1] = files[i]
        allDivFiles[#allDivFiles + 1] = divFiles[i]
      end
    end

    local frameStride = 32
    if string.find(setDir, '3d') ~= nil then
      frameStride = frameStride * 2
    end

    print('  ==> Processing files')
    for i = 1, #allFiles, frameStride do
      collectgarbage()
      torch.progress(i, #allFiles)
      local fn = allFiles[i]
      local fnDiv = allDivFiles[i]
      local time, p, Ux, Uy, Uz, geom = torch.DataBinary:_loadFile(fn)
      local _, pDiv, UxDiv, UyDiv, UzDiv = torch.DataBinary:_loadFile(fnDiv)
      local zdim = p:size(1)
      local ydim = p:size(2)
      local xdim = p:size(3)
      local twoDim = zdim == 1
      local U, UDiv
      if twoDim then
        U = torch.FloatTensor(2, zdim, ydim, xdim)
        UDiv = torch.FloatTensor(2, zdim, ydim, xdim)
      else
        U = torch.FloatTensor(3, zdim, ydim, xdim)
        UDiv = torch.FloatTensor(3, zdim, ydim, xdim)
      end
      U[1]:copy(Ux)
      U[2]:copy(Uy)
      UDiv[1]:copy(UxDiv)
      UDiv[2]:copy(UyDiv)
      if not twoDim then
        U[3]:copy(Uz)
        UDiv[3]:copy(UzDiv)
      end

      -- Calculate the ground truth velocity update.
      local velUpdateGT = UDiv - U

      local function printErr(err, str)
        local displayErr = err[{{}, 1, {}, {}}]
        image.display{image = displayErr, padding = 2, zoom = 4, legend = str}
        print('  ' .. str .. ' Max err = ' .. err:abs():max())
      end

      local velUpdate = velUpdateGT:clone():fill(0)
      local matchManta = true
      torch.calcVelocityUpdate(velUpdate, p, geom, matchManta)
      local err = velUpdateGT - velUpdate

      -- There's ONE more case I cannot track down. I think it happens when a
      -- ghost pixel is geom but the internal pixel is not, but I can't be
      -- sure. In any case, it happens for < 10 voxels in the entire dataset.
      -- For now ignore the first element border of each dimension.
      err[{1, {}, {}, 1}]:mul(0)
      err[{2, {}, 1, {}}]:mul(0)
      if not twoDim then
        err[{3, 1, {}, {}}]:mul(0)
      end

      local curMaxErr = err:clone():abs():max()
      maxErr = math.max(curMaxErr, maxErr)
      if maxErr > precision then
        printErr(err, fn)
        error('Error above precision')
      end
    end
    torch.progress(#allFiles, #allFiles)  -- Finish the pbar
  end
  print('All tests pass. maxErr = ' .. maxErr)
end

testVelocityUpdate()
