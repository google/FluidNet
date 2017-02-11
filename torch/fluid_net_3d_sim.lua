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

-- Simulation script to run a 3D model for a particular scene and dump the
-- results to file (for rendering in Blender).
--
-- We give a single boundary condition example of a plume from the bottom of
-- the grid with otherwise open boundary conditions.
--
-- The model in the scene is controlled through the 'loadVoxelModel'
-- parameter.
--
-- The output is a sequence of .vbox files in the CNNFluids/blender folder.
-- These files are then loaded into blender for rendering.
--
-- Typical useage:
-- qlua fluid_net_3d_sim.lua -modelFilename myModel -loadVoxelModel arc \
--                           -visualizeData true -saveData true

local tfluids = require('tfluids')
local paths = require('paths')
dofile("lib/include.lua")
dofile("lib/save_parameters.lua")
dofile("lib/obstacles_import_binvox.lua")

-- ****************************** Define Config ********************************
local conf = torch.defaultConf()
conf.batchSize = 1
conf.loadModel = true
conf.visualizeData = true
conf.saveData = true
conf.densityFilename = ''  -- empty string ==> use default path.
conf.plumeSimMethod = 'convnet'  -- options are: 'convnet', 'jacobi', 'pcg'
conf = torch.parseArgs(conf)  -- Overwrite conf params from the command line.
assert(conf.batchSize == 1, 'The batch size must be one')
assert(conf.loadModel == true, 'You must load a pre-trained model')

-- ****************************** Select the GPU *******************************
cutorch.setDevice(conf.gpu)
print("GPU That will be used:")
print(cutorch.getDeviceProperties(conf.gpu))

-- ***************************** Create the model ******************************
conf.modelDirname = conf.modelDir .. '/' .. conf.modelFilename
local mconf, model = torch.loadModel(conf.modelDirname)
model:cuda()
torch.setDropoutTrain(model, false)
assert(mconf.is3D, 'The model must be 3D')

-- *************************** Define some variables ***************************
local res = 128  -- Choose power of 2 from 16 to 512.
local batchCPU = {
    pDiv = torch.FloatTensor(conf.batchSize, 1, res, res, res):fill(0),
    UDiv = torch.FloatTensor(conf.batchSize, 3, res, res, res):fill(0),
    flags = tfluids.emptyDomain(torch.FloatTensor(
        conf.batchSize, 1, res, res, res), true),
    density = torch.FloatTensor(conf.batchSize, 1, res, res, res):fill(0)
}
print("running simulation at resolution " .. res .. "^3")

-- **************** Set some nice looking simulation variables  ****************
mconf.buoyancyScale = 2.0 * (res / 128)  -- 1 is good for 'none', 2 otherwise.
mconf.gravityScale = 0
mconf.dt = 0.1
local plumeScale = 1.0 * (res / 128)
local numFrames = 768
local outputDecimation = 3
mconf.maccormackStrength = 0.6
mconf.maxIter = 34  -- For jacobi or pcg only (34 matches our model at 128).
-- Option 1:
mconf.vorticityConfinementAmp = 3  -- Comparison videos rendered with 0
mconf.advectionMethod = 'maccormackOurs'
-- Option 2:
-- mconf.vorticityConfinementAmp = 2
-- mconf.advectionMethod = 'eulerOurs'
mconf.simMethod = conf.plumeSimMethod

-- *************************** Load a model into flags *************************
local voxels = {}
local outDir
if conf.loadVoxelModel ~= "none" then
  -- We want the model resolution to be HALF the grid resolution.
  local modelRes = math.pow(2, math.floor(math.log(res) / math.log(2)) - 1)
  local offsetX = 0
  local offsetY = 0
  local offsetZ = 0
  print('Using model resolution: ' .. modelRes)
  if conf.loadVoxelModel == "arch" then
    voxels = tfluids.loadVoxelData('../voxelizer/voxels_demo/Y91_arc_' ..
                                   modelRes .. '.binvox')
    --This lines up the arc correctly
    tfluids.flipDiagonal(voxels.data, 2)
    tfluids.flipDiagonal(voxels.data, 0)
    outDir = '../blender/arch_render/'
    offsetY = -0.04 * res
  elseif conf.loadVoxelModel == "bunny" then
    voxels = tfluids.loadVoxelData(
      '../voxelizer/voxels_demo/bunny.capped_' .. modelRes .. '.binvox')
    tfluids.flipDiagonal(voxels.data, 2)
    tfluids.flipDiagonal(voxels.data, 0)
    outDir = '../blender/bunny_render/'
    offsetX = 0.04 * res  -- The center of the base is not center of BBOX.
    offsetZ = 0.04 * res
  else
    error('Bad conf.loadVoxelModel value')
  end
  voxels.data = tfluids.padVoxelsToDims(res, res, res, voxels.data,
                                        offsetX, offsetY, offsetZ)

  -- We need to turn the occupancy grid {0, 1} into a proper flags grid.
  local occ = voxels.data:view(1, 1, res, res, res)
  -- We need to copy in the occupancy data, but ONLY within the 1 pix / vox
  -- border.
  occ = occ[{{}, {}, {2, res - 1}, {2, res - 1}, {2, res - 1}}]:contiguous()
  local flagsInBounds =
      batchCPU.flags[{{}, {}, {2, res - 1}, {2, res - 1}, {2, res - 1}}]
  flagsInBounds:copy((occ * tfluids.CellType.TypeObstacle) +
                     ((1 - occ) * tfluids.CellType.TypeFluid))
else
  outDir = '../blender/mushroom_cloud_render/'
end
--*****************************************************************************
local batchGPU = {}
for key, value in pairs(batchCPU) do
  batchGPU[key] = value:cuda()
end
local frameCounter = 1

local simulationTimeSec = numFrames * mconf.dt
print('Simulating with dt = ' .. mconf.dt)
print('Saving every ' .. outputDecimation .. ' frames')
print('Simulating for ' .. numFrames .. ' frames (' .. simulationTimeSec ..
      'sec)')
print('    mconf:')
print(torch.tableToString(mconf))

-- ****************************** DATA FUNCTIONS *******************************
-- Set up a plume boundary condition.
local densityVal = {1}
local rad = 0.15
tfluids.createPlumeBCs(batchGPU, densityVal, plumeScale, rad)

-- ***************************** Create Voxel File ****************************
local densityFile, densityFilename, obstaclesFile, obstaclesFilename
local obstaclesBlenderFile, obstaclesBlenderFilename
if conf.saveData then
  if string.len(conf.densityFilename) < 1 then
    densityFilename = (outDir .. '/density_output_' .. conf.modelFilename ..
                       '_dt' .. mconf.dt .. '.vbox')
  else
    densityFilename = conf.densityFilename
  end
  densityFile = torch.DiskFile(densityFilename,'w')
  densityFile:binary()
  densityFile:writeInt(res)
  densityFile:writeInt(res)
  densityFile:writeInt(res)
  densityFile:writeInt(numFrames)
  
  obstaclesFilename = (outDir .. '/geom_output.vbox')
  obstaclesFile = torch.DiskFile(obstaclesFilename,'w')
  obstaclesFile:binary()
  obstaclesFile:writeInt(res)
  obstaclesFile:writeInt(res)
  obstaclesFile:writeInt(res)
  obstaclesFile:writeInt(1)

  obstaclesBlenderFilename = (outDir .. '/geom_output_blender.vbox')
  obstaclesBlenderFile = torch.DiskFile(obstaclesBlenderFilename,'w')
  obstaclesBlenderFile:binary()
  obstaclesBlenderFile:writeInt(res)
  obstaclesBlenderFile:writeInt(res)
  obstaclesBlenderFile:writeInt(res)
  obstaclesBlenderFile:writeInt(1)

  print('Writing density to: ' .. densityFilename)
  print('Writing geom to: ' .. obstaclesFilename)
  print('Writing blender geom to: ' .. obstaclesBlenderFilename)
end

local divNet = tfluids.VelocityDivergence():cuda()

function visualizeData(batchGPU, hImage)
  -- This is SUPER slow. We're pulling off the GPU and then sending it back
  -- as an image.display call. It probably limits framerate.
  local _, U, flags, density, _ = tfluids.getPUFlagsDensityReference(batchGPU)
  local div = divNet:forward({U, flags})
  div = div:squeeze():mean(1):squeeze()  -- Mean along z-channel.
  local flags = flags:squeeze():mean(1):squeeze()
  local density = density:mean(2):squeeze():sqrt()
  local densityz = density:mean(1):squeeze()  -- Average along Z dimension.
  local densityx = density:mean(3):squeeze():t():contiguous()
  local sz = {1, 1, div:size(1), div:size(2)}

  flags = flags:view(unpack(sz)):float()
  div = div:view(unpack(sz)):float()
  densityx = densityx:view(unpack(sz)):float()
  densityz = densityz:view(unpack(sz)):float()
  local im = torch.cat({flags, div, densityz, densityx}, 1)
  im = image.flip(im, 3)  -- Flip y dimension.
  if hImage == nil then
    hImage = image.display{image = im, zoom = 512 / density:size(1),
                           gui = false, padding = 2, scaleeach = true, nrow = 2,
                           legend = 'flags, div, density_z, density_x'}
  else
    image.display{image = im, zoom = 512 / density:size(1),
                  gui = false, padding = 2, scaleeach = true, nrow = 2,
                  legend = 'flags, div, density_z, density_x', win = hImage}
  end
  return hImage
end

local hImage
if conf.visualizeData then
  hImage = visualizeData(batchGPU)
end

-- ***************************** SIMULATION LOOP *******************************
local obstacles = batchGPU.flags:clone()
local t0
local msgAccum = 0
tfluids.profilePressure = true  -- Enable profiling.
for i = 1, numFrames do
  local curFps, dt
  if i == 2 then
    t0 = sys.clock()  -- Don't include timing for the first frame.
  end
  if i > 2 then
    dt = sys.clock() - t0
    curFps = (i - 2) / dt
  end

  if math.fmod(i, 20) == 0 then
    collectgarbage()
  end

  if i <= 2 or (dt - msgAccum > 1) then
    msgAccum = dt or 0
    local msg = 'Simulating frame ' .. i .. ' of ' .. numFrames
    if curFps ~= nil then
      local sec = (numFrames - i) / curFps
      local min = math.floor(sec / 60)
      sec = sec - (min * 60)
      msg = string.format('%s (est time remaining %d:%02d min:sec)',
                          msg, min, sec)
    end
    print(msg)
  end
  
  tfluids.simulate(conf, mconf, batchGPU, model, false)
  -- Result is now on the GPU.

  local p, U, flags, density = tfluids.getPUFlagsDensityReference(batchGPU)

  if conf.saveData then
    if i == 1 then
      -- Convert flags to obstacles array with 0, 1 for occupied.
      -- The next call assumes that the domain is fluid everywhere else and that
      -- there are no obstacle inflow regions.
      tfluids.flagsToOccupancy(flags, obstacles)

      obstaclesFile:writeFloat(
          obstacles:squeeze():permute(3, 2, 1):float():contiguous():storage())

      -- Zero out the bnd obstacles (otherwise when we render it we wont be able
      -- to see anything).
      for dim = 3, 5 do
        obstacles:select(dim, 1):fill(0)
        obstacles:select(dim, res):fill(0)
      end
      obstaclesBlenderFile:writeFloat(
          obstacles:squeeze():permute(3, 2, 1):float():contiguous():storage())
    end

    if math.fmod(i, outputDecimation) == 0 then
      -- Save greyscale density (so mean across RGB).
      densityFile:writeFloat(density:mean(2):squeeze():permute(
          3, 2, 1):float():contiguous():storage())
    end
  end

  if conf.visualizeData then
    visualizeData(batchGPU, hImage)
  end
end
cutorch.synchronize()
local t1 = sys.clock()
print('All done!')
print('Processing time: ' .. (1000 * (t1 - t0) / (numFrames - 1)) ..
      ' ms per frame')
print('Processing time linear projection: ' .. 
      (1000 * tfluids.profilePressureTime / tfluids.profilePressureCount) ..
      ' ms per frame')


if conf.saveData then
  densityFile:close()
  obstaclesFile:close()
end

-- Close the image in case we're running the script in a batch.
if conf.visualizeData then
  hImage:close()
end

