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
print('==> Loaded model from: ' .. conf.modelDirname)
torch.setDropoutTrain(model, false)
assert(mconf.is3D, 'The model must be 3D')

-- *************************** Define some variables ***************************
local res = 128
local batchCPU = {
    pDiv = torch.FloatTensor(conf.batchSize, 1, res, res, res):fill(0),
    UDiv = torch.FloatTensor(conf.batchSize, 3, res, res, res):fill(0),
    flags = tfluids.emptyDomain(torch.FloatTensor(
        conf.batchSize, 1, res, res, res), true),
    density = torch.FloatTensor(conf.batchSize, 1, res, res, res):fill(0)
}
print("running simulation at resolution " .. res .. "^3")

-- **************** Set some nice looking simulation variables  ****************
mconf.buoyancyScale = 2.0
mconf.dt = 0.1
local plumeScale = 1.0
local numFrames = 768
local outputDecimation = 3
mconf.maccormackStrength = 0.8
-- Option 1:
mconf.vorticityConfinementAmp = 0
mconf.advectionMethod = 'maccormackOurs'
-- Option 2:
-- mconf.vorticityConfinementAmp = tfluids.getDx(batchCPU.flags) * 4
-- mconf.advectionMethod = 'rk2Ours'
mconf.simMethod = conf.plumeSimMethod

-- *************************** Load a model into flags *************************
local voxels = {}
local outDir
if conf.loadVoxelModel ~= "none" then
  if conf.loadVoxelModel == "arc" then
    voxels = tfluids.loadVoxelData('../voxelizer/voxels_demo/Y91_arc_64.binvox')
    --This lines up the arc correctly
    tfluids.flipDiagonal(voxels.data, 2)
    tfluids.flipDiagonal(voxels.data, 0)
    outDir = '../blender/arch_render/'
  elseif conf.loadVoxelModel == "bunny" then
    voxels = tfluids.loadVoxelData(
      '../voxelizer/voxels_demo/bunny.capped_64.binvox')
    outDir = '../blender/bunny_render/'
  else
    error('Bad conf.loadVoxelModel value')
  end
  voxels.data = tfluids.expandVoxelsToDims(res, res, res, voxels.data)
  tfluids.moveVoxelCentroidToCenter(voxels.data)
  voxels.dims = {res, res, res}
  local bb = tfluids.calculateBoundingBox(voxels.data)
  voxels.min = bb.min
  voxels.max = bb.max

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
end

local divNet = tfluids.VelocityDivergence():cuda()

local hImage
if conf.visualizeData then
  local _, U, flags, _ = tfluids.getPUFlagsDensityReference(batchGPU)

  local div = divNet:forward({U, flags})
  div = div:squeeze():mean(1):squeeze()  -- Mean along z-channel.

  local density = batchGPU.density:mean(2):squeeze()
  density = density:mean(1):squeeze()  -- Average along Z dimension.

  local sz = {1, div:size(1), div:size(2)}
  local im = torch.cat(div:view(unpack(sz)), density:view(unpack(sz)), 1)
  hImage = image.display{image = im, zoom = 512 / density:size(1),
                         gui = false, legend = 'divergence & density',
                         padding = 2}
end

-- ***************************** SIMULATION LOOP *******************************
local obstacles = batchGPU.flags:clone()
sys.tic()
for i = 1, numFrames do
  if math.fmod(i, 20) == 0 then
    collectgarbage()
  end
  print('Simulating frame ' .. i .. ' of ' .. numFrames)
  
  tfluids.simulate(conf, mconf, batchGPU, model, false)
  -- Result is now on the GPU.

  local p, U, flags, density = tfluids.getPUFlagsDensityReference(batchGPU)

  if conf.saveData then
    -- Convert flags to obstacles array with 0, 1 for occupied.
    -- The next call assumes that the domain is fluid everywhere else and that
    -- there are no obstacle inflow regions.
    tfluids.flagsToOccupancy(flags, obstacles)

    if i == 1 then
      obstaclesFile:writeFloat(
          obstacles:squeeze():permute(3, 2, 1):float():contiguous():storage())
      print('  ==> Saved obstacles to ' .. obstaclesFilename)
    end
    if math.fmod(i, outputDecimation) == 0 then
      -- Save greyscale density (so mean across RGB).
      densityFile:writeFloat(density:mean(2):squeeze():permute(
          3, 2, 1):float():contiguous():storage())
      print('  ==> Saved density to ' .. densityFilename)
    end
  end

  if conf.visualizeData then
    -- This is SUPER slow. We're pulling off the GPU and then sending it back
    -- as an image.display call. It probably limits framerate.
    local div = divNet:forward({U, flags})
    div = div:squeeze():mean(1):squeeze()  -- Mean along z-channel.

    local density = batchGPU.density:mean(2):squeeze()  -- Mean of RGB
    density = density:mean(1):squeeze():sqrt()  -- Mean along z-channel.

    local sz = {1, div:size(1), div:size(2)}
    local im = torch.cat(div:view(unpack(sz)), density:view(unpack(sz)), 1)
    image.display{image = im, zoom = 512 / density:size(1),
                  gui = false, legend = 'divergence & density', win = hImage,
                  padding = 2}
  end
end
cutorch.synchronize()
local t = sys.toc()
print('All done!')
print('Processing time: ' .. (1000 * t / numFrames) .. ' ms per frame')

if conf.saveData then
  densityFile:close()
  obstaclesFile:close()
end

-- Close the image in case we're running the script in a batch.
if conf.visualizeData then
  hImage:close()
end

