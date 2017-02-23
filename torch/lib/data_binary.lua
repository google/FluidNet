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

-- Top level data processing code.
--
-- TODO(tompson): This needs a huge cleanup.

local mattorch = torch.loadPackageSafe('mattorch')
local torchzlib = torch.loadPackageSafe('torchzlib')
local torchzfp = torch.loadPackageSafe('torchzfp')
local paths = require('paths')
local tfluids = require('tfluids')

local DataBinary, parent = torch.class('torch.DataBinary')

local COMPRESS_DATA_ON_DISK = false
local ZFP_ACCURACY = 1e-5
local DIV_THRESHOLD = 1e-2  -- Relaxed threshold.

function DataBinary:__init(conf, prefix)
  -- Load the processed data.
  self.dataDir = conf.dataDir  -- Just for reference
  self.dataset = conf.dataset  -- Just for reference
  self.prefix = prefix
  local baseDir = self:_getBaseDir(conf.dataDir) .. '/'
  local runDirs = torch.ls(baseDir)
  assert(#runDirs > 0, "couldn't find any run directories in " .. baseDir)

  -- Create a network to calculate divergence of the set (see comment below).
  local divNet = tfluids.VelocityDivergence()

  -- Since the flat files array will include runs from multiple simulations, we
  -- need a list of valid frame pairs (to avoid trying to predict the transition
  -- between the last frame in one run and the first in another).
  self.runs = {}  -- Not necessarily the entire set (we may remove some).
  self.minValue = math.huge
  for i = 1, #runDirs do
    torch.progress(i, #runDirs)
    local runDir = self:_getBaseDir(conf.dataDir) .. '/' .. runDirs[i] .. '/'
    local files = torch.ls(runDir .. '*[0-9].bin')
    local divFiles = torch.ls(runDir .. '*[0-9]_divergent.bin')

    assert(#files == #divFiles,
           "Main file count not equal to divergence file count")
    if #files <= conf.ignoreFrames then
      print('WARNING: Not enough files in sub-dir ' .. runDir)
      print('  skipping it')
    else
      -- Ignore the first n frames.
      local runFiles = {}
      local runFilesDivergence = {}
      for f = conf.ignoreFrames + 1, #files do
        runFiles[#runFiles + 1] = files[f]
        runFilesDivergence[#runFilesDivergence + 1] = divFiles[f]
      end
      assert(#runFiles == #runFilesDivergence)
  
      local data = {}
      for f = 1, #runFiles do
        collectgarbage()
        local p, U, flags, density, is3D =
            torch.loadMantaFile(runFiles[f])
        local pDiv, UDiv, flagsDiv, densityDiv, is3DDiv =
            torch.loadMantaFile(runFilesDivergence[f])
        -- The flags shouldn't change.
        assert(torch.all(torch.eq(flags, flagsDiv)), 'ERROR: Flags changed!')
        assert(is3D == is3DDiv, '3D flag is inconsistent!')
  
        if data.p == nil then
          -- Initialize the data structure.
          local xdim = p:size(5)
          local ydim = p:size(4)
          local zdim = p:size(3)
          local nuchans = U:size(2)
          assert(UDiv:size(2) == nuchans)
          if not is3D then
            assert(zdim == 1, '2D domains should have unary z-dimension')
            assert(nuchans == 2, '2D domains should have 2D velocity')
          else
            assert(nuchans == 3, '3D domains should have 3D velocity')
          end
          data.p = torch.FloatTensor(#runFiles, 1, zdim, ydim, xdim)
          data.flags = torch.FloatTensor(#runFiles, 1, zdim, ydim, xdim)
          data.U = torch.FloatTensor(#runFiles, nuchans, zdim, ydim, xdim)
          data.density = torch.FloatTensor(#runFiles, 1, zdim, ydim, xdim)
          data.pDiv = torch.FloatTensor(#runFiles, 1, zdim, ydim, xdim)
          data.UDiv = torch.FloatTensor(#runFiles, nuchans, zdim, ydim, xdim)
          data.densityDiv = torch.FloatTensor(#runFiles, 1, zdim, ydim, xdim)
          data.is3D = is3D
        else
          assert(is3D == data.is3D, 'inconsistent 3D flags across run files')
        end
  
        -- We could be pedantic here and start checking that the sizes of all
        -- files match, but instead just let the copy fail if numel doesn't
        -- match and assume the dims do.
        data.p[f]:copy(p)
        data.density[f]:copy(density)
        data.flags[f]:copy(flags)
        data.U[f]:copy(U)
        data.pDiv[f]:copy(pDiv)
        data.UDiv[f]:copy(UDiv)
        data.densityDiv[f]:copy(densityDiv)
      end
  
      local xdim = data.p:size(5)
      local ydim = data.p:size(4)
      local zdim = data.p:size(3)
      local is3D = data.is3D
  
      -- Unfortunately, some samples might be the result of an unstable Manta
      -- sim. i.e. the divergence blew up, but not completely. If this is the
      -- case, then we should ignore the entire run. So we need to check the
      -- divergence of the target velocity.
      local divTarget = divNet:forward({data.U, data.flags})
      local maxDiv = divTarget:max()
      if maxDiv > DIV_THRESHOLD then
        print('WARNING: run ' .. runDirs[i] .. ' (i = ' .. i ..
              ') has a sample with max(div) = ' .. maxDiv ..
              ' which is above the allowable threshold (' .. DIV_THRESHOLD ..
              ')')
        print('  --> Removing run from the dataset...')
      else
        -- This sample is OK.
        
        -- For 3D data it's too much data to store, so we have to catch the
        -- data on disk and read it back in on demand. This then means we'll
        -- need an asynchronous mechanism for loading data to hide the load
        -- latency.
        self:_cacheDataToDisk(data, conf, runDirs[i])
        data = nil  -- No longer need it (it's cached on disk).
  
        self.runs[#self.runs + 1] = {
          dir = runDirs[i],
          ntimesteps = #runFiles,
          xdim = xdim,
          ydim = ydim,
          zdim = zdim,
          is3D = is3D
        }
      end
    end
  end
  runDirs = nil  -- Full set no longer needed.

  -- Check that the size of each run is the same.
  self.xdim = self.runs[1].xdim
  self.ydim = self.runs[1].ydim
  self.zdim = self.runs[1].zdim
  self.is3D = self.runs[1].is3D
  for i = 2, #self.runs do
    assert(self.xdim == self.runs[i].xdim)
    assert(self.ydim == self.runs[i].ydim)
    assert(self.zdim == self.runs[i].zdim)
    assert(self.is3D == self.runs[i].is3D)
  end

  -- Create an explicit list of trainable frames.
  self.samples = {}
  for r = 1, #self.runs do
    for f = 1, self.runs[r].ntimesteps  do
      self.samples[#self.samples + 1] = {r, f}
    end
  end
  self.samples = torch.LongTensor(self.samples)
end

function DataBinary:_getBaseDir(dataDir)
  return dataDir .. '/' .. self.dataset .. '/' .. self.prefix .. '/'
end

-- Note: runDir is typically self.runs[irun].dir
function DataBinary:_getCachePath(dataDir, runDir, iframe)
  local dir = self:_getBaseDir(dataDir) .. '/' .. runDir .. '/'
  local fn = string.format('frame_cache_%06d', iframe)
  return dir .. fn
end

function DataBinary:_cacheDataToDisk(data, conf, runDir)
  -- The function will take in a table of tensors and an output directory
  -- and store the result on disk.
  local is3D = data.is3D
  data.is3D = nil

  -- Firstly, make sure that each tensor in data has the same size(1).
  local nframes
  for _, value in pairs(data) do
    assert(torch.isTensor(value) and torch.type(value) == 'torch.FloatTensor')
    if nframes == nil then
      nframes = value:size(1)
    else
      assert(value:size(1) == nframes)
    end
  end

  -- Now store each frame on disk in a separate file.
  for iframe = 1, nframes do
    local frameData = {}
    for key, value in pairs(data) do
      local v = value[{iframe}]
      -- Annoyingly, if we slice the tensor, torch.save will still save the
      -- entire storage (i.e. all frames). We need to clone the tensor
      -- to prevent this.
      if torch.isTensor(v) then
        v = v:clone()
      end
      if COMPRESS_DATA_ON_DISK and torch.isTensor(v) and v:dim() > 1 then
        v = torch.ZFPTensor(v, ZFP_ACCURACY)
      end
      frameData[key] = v
    end
    frameData.is3D = is3D
    torch.save(self:_getCachePath(conf.dataDir, runDir, iframe), frameData)
  end

  data.is3D = is3D
end

function DataBinary:_loadDiskCache(dataDir, irun, iframe)
  assert(irun <= #self.runs and irun >= 1)
  assert(iframe <= self.runs[irun].ntimesteps and iframe >= 1)
  local fn = self:_getCachePath(dataDir, self.runs[irun].dir, iframe)
  local data = torch.load(fn)
  for key, value in pairs(data) do
    if torch.type(value) == 'torch.ZFPTensor' then
      data[key] = value:decompress()
    end
  end
  assert(data.is3D == self.is3D)  -- Just to be sure.
  return data
end

-- Abstract away getting data from however we store it.
function DataBinary:getSample(dataDir, irun, iframe)
  assert(irun <= #self.runs and irun >= 1, 'getSample: irun out of bounds!')
  assert(iframe <= self.runs[irun].ntimesteps and iframe >= 1,
    'getSample: iframe out of bounds!')

  return self:_loadDiskCache(dataDir, irun, iframe)
end

function DataBinary:saveSampleToMatlab(conf, filename, irun, iframe)
  local s = self:getSample(conf.dataDir, irun, iframe)
  local data = {p = s.p, U = s.U, flags = s.flags, density = s.density,
                pDiv = s.pDiv, UDiv = s.UDiv, densityDiv = s.densityDiv}
  for key, value in pairs(data) do
    data[key] = value:double():squeeze()
  end
  assert(mattorch ~= nil, 'saveSampleToMatlab requires mattorch.')
  mattorch.save(filename, data)
end

-- Visualize the components of a concatenated tensor of (p, U, flags).
function DataBinary:_visualizeBatchData(data, legend, depth)
  -- Check the input.
  assert(data.p ~= nil and data.flags ~= nil and data.U ~= nil)

  legend = legend or ''
  depth = depth or math.ceil(data.U:size(3) / 2)
  local nrow = math.floor(math.sqrt(data.flags:size(1)))
  local zoom = 1024 / (nrow * self.xdim)
  zoom = math.min(zoom, 2)
  local is3D = data.U:size(2) == 3

  local p, Ux, Uy, flags, density

  -- Just visualize one depth slice.
  p = data.p:float():select(3, depth)
  Ux = data.U[{{}, {1}}]:float():select(3, depth)
  Uy = data.U[{{}, {2}}]:float():select(3, depth)
  flags = data.flags:float():select(3, depth)
  if (data.density ~= nil) then
    density = data.density:float():select(3, depth)
  end

  local scaleeach = false;
  image.display{image = p, zoom = zoom, padding = 2, nrow = nrow,
    legend = (legend .. ': p'), scaleeach = scaleeach}
  image.display{image = Ux, zoom = zoom, padding = 2, nrow = nrow,
    legend = (legend .. ': Ux'), scaleeach = scaleeach}
  image.display{image = Uy, zoom = zoom, padding = 2, nrow = nrow,
    legend = (legend .. ': Uy'), scaleeach = scaleeach}
  image.display{image = flags, zoom = zoom, padding = 2, nrow = nrow,
    legend = (legend .. ': flags'), scaleeach = scaleeach}
  if (density ~= nil) then
    image.display{image = density, zoom = zoom, padding = 2, nrow = nrow,
      legend = (legend .. ': density'), scaleeach = scaleeach}
  end
end

function DataBinary:visualizeBatch(conf, mconf, imgList, depth)
  if imgList == nil then
    local shuffle = torch.randperm(self:nsamples())
    imgList = {}
    for i = 1, conf.batchSize do
      table.insert(imgList, shuffle[{i}])
    end
  end

  local batchCPU = self:AllocateBatchMemory(conf.batchSize)

  depth = depth or math.ceil(batchCPU.pDiv:size(3) / 2)

  self:CreateBatch(batchCPU, torch.IntTensor(imgList), conf.batchSize,
                   conf.dataDir)
  local batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')
  -- Also call a sync just to test that as well.
  torch.syncBatchToGPU(batchCPU, batchGPU)

  print('    Image set:')  -- Don't print if this is used in the train loop.
  for i = 1, #imgList do
    local isample = imgList[i]

    local irun = self.samples[isample][1]
    local iframe = self.samples[isample][2]

    print(string.format("    %d - sample %d: irun = %d, iframe = %d", i,
                        imgList[i], irun, iframe))
  end

  local range = {1, #imgList}
  for key, value in pairs(batchGPU) do
    batchGPU[key] = value[{range}]  -- Just pick the samples in the imgList.
  end

  local inputData = {
      p = batchGPU.pDiv,
      U = batchGPU.UDiv,
      flags = batchGPU.flags,
      density = batchGPU.densityDiv
  }
  self:_visualizeBatchData(inputData, 'input', depth)

  local outputData = {
      p = batchGPU.pTarget,
      U = batchGPU.UTarget,
      flags = batchGPU.flags,
      density = batchGPU.densityTarget
  }
  self:_visualizeBatchData(outputData, 'target', depth)

  return batchCPU, batchGPU
end

function DataBinary:AllocateBatchMemory(batchSize, ...)
  collectgarbage()  -- Clean up thread memory.
  -- Create the data containers.
  local d = self.zdim
  local h = self.ydim
  local w = self.xdim

  local batchCPU = {}

  local numUChans = 3
  if not self.is3D then
    numUChans = 2
  end
  batchCPU.pDiv = torch.FloatTensor(batchSize, 1, d, h, w):fill(0)
  batchCPU.pTarget = batchCPU.pDiv:clone()
  batchCPU.UDiv = torch.FloatTensor(batchSize, numUChans, d, h, w):fill(0)
  batchCPU.UTarget = batchCPU.UDiv:clone()
  batchCPU.flags = tfluids.emptyDomain(torch.FloatTensor(batchSize, 1, d, h, w),
                                       self.is3D)
  batchCPU.densityDiv = torch.FloatTensor(batchSize, 1, d, h, w):fill(0)
  batchCPU.densityTarget = batchCPU.densityDiv:clone()

  return batchCPU
end

local function downsampleImagesBy2(dst, src)
  assert(src:dim() == 4)
  local h = src:size(3)
  local w = src:size(4)
  local nchans = src:size(2)
  local nimgs = src:size(1)

  -- EDIT: This is OK for CNNFluids.  We pad at the output of the bank to
  -- resize the output tensor
  -- assert(math.mod(w, 2) == 0 and math.mod(h, 2) == 0,
  --   'bank resolution is not a power of two divisible by its parent')

  assert(dst:size(1) == nimgs and dst:size(2) == nchans and
    dst:size(3) == math.floor(h / 2) and dst:size(4) == math.floor(w / 2))

  -- Convert src to 3D tensor
  src:resize(nimgs * nchans, h, w)
  dst:resize(nimgs * nchans, math.floor(h / 2), math.floor(w / 2))
  -- Perform the downsampling
  src.image.scaleBilinear(src, dst)
  -- Convert src back to 4D tensor
  src:resize(nimgs, nchans, h, w)
  dst:resize(nimgs, nchans, math.floor(h / 2), math.floor(w / 2))
end

-- CreateBatch fills already preallocated data structures.  To be used in a
-- training loop where the memory is allocated once and reused.
-- @param batchCPU - batch CPU table container.
-- @param sampleSet - 1D tensor of sample indices.
-- @param ... - 2 arguments:
--     batchSize - size of batch container (usually conf.batchSize).
--     dataDir - path to data directory cache (usually conf.dataDir).
function DataBinary:CreateBatch(batchCPU, sampleSet, ...)
  -- Unpack variable length args.
  local args = {...}

  assert(#args == 2, 'Expected 2 additional args to CreateBatch')
  local batchSize = args[1]  -- size of batch container, not the sampleSet.
  local dataDir = args[2]

  assert(sampleSet:size(1) <= batchSize)  -- sanity check.

  if perturb == nil then
    perturb = false
  end

  -- Make sure we haven't done anything silly.
  local is3D = batchCPU.UDiv:size(2) == 3
  assert(self.is3D == is3D, 'Mixing 3D model with 2D data or vice-versa!')

  -- create mini batch
  for i = 1, sampleSet:size(1) do
    -- For each image in the batch list
    local isample = sampleSet[i]
    assert(isample >= 1 and isample <= self:nsamples())
    local irun = self.samples[isample][1]
    local iframe = self.samples[isample][2]

    -- Get the sample data.
    local sample = self:getSample(dataDir, irun, iframe)

    -- Copy the input channels to the batch array.
    batchCPU.flags[i]:copy(sample.flags)
    batchCPU.pDiv[i]:copy(sample.pDiv)
    batchCPU.UDiv[i]:copy(sample.UDiv)
    batchCPU.densityDiv[i]:copy(sample.densityDiv)

    -- Target is the corrected velocity (and corresponding pressure) field.
    batchCPU.pTarget[i]:copy(sample.p)
    batchCPU.UTarget[i]:copy(sample.U)
    batchCPU.densityTarget[i]:copy(sample.density)
  end
end

function DataBinary:nsamples()
  -- Return the maximum number of image pairs we can support
  return self.samples:size(1)
end

function DataBinary:initThreadFunc()
  -- Recall: All threads must be initialized with all classes and packages.
  dofile('lib/fix_file_references.lua')
  dofile('lib/load_package_safe.lua')
  dofile('lib/data_binary.lua')
end

function DataBinary:calcDataStatistics(conf, mconf)
  -- Calculate the mean and std of the input channels for each sample, and
  -- plot it as a histogram.
  
  -- Parallelize data loading to increase disk IO.
  local singleThreaded = false  -- Set to true for easier debugging.
  local numThreads = 8
  local batchSize = 16
  local dataInds = torch.IntTensor(torch.range(1, self:nsamples()))
  local parallel = torch.DataParallel(conf.numDataThreads, self, dataInds,
                                      conf.batchSize, DataBinary.initThreadFunc,
                                      singleThreaded)

  -- We also want to calculate the divergence of the (divergent) U channel
  -- input. This is because some models actually use this as input.
  local divNet = tfluids.VelocityDivergence()

  local mean = {}
  local std = {}
  local l2 = {}
  print('Calculating mean and std statistics.')
  local samplesProcessed = 0
  repeat
    -- Get the next batch.
    local batch = parallel:getBatch(conf.batchSize, conf.dataDir)
    local batchCPU = batch.data
    batchCPU.div = divNet:forward({batchCPU.UDiv, batchCPU.flags})

    for key, value in pairs(batchCPU) do
      local value1D = value:view(value:size(1), -1)  -- reshape [bs, -1]
      local curMean = torch.mean(value1D, 2):squeeze()
      local curStd = torch.std(value1D, 2):squeeze()
      local curL2 = torch.norm(value1D, 2, 2):squeeze()
      if mean[key] == nil then
        mean[key] = {}
      end
      if std[key] == nil then
        std[key] = {}
      end
      if l2[key] == nil then
        l2[key] = {}
      end
      for i = 1, batch.batchSet:size(1) do
        mean[key][#(mean[key]) + 1] = curMean[i]
        std[key][#(std[key]) + 1] = curStd[i]
        l2[key][#(l2[key]) + 1] = curL2[i]
      end
    end
    samplesProcessed = samplesProcessed + batch.batchSet:size(1)
    torch.progress(samplesProcessed, dataInds:size(1))
  until parallel:empty()

  for key, value in pairs(mean) do
    mean[key] = torch.FloatTensor(value)
  end
  for key, value in pairs(std) do
    std[key] = torch.FloatTensor(value)
  end
  for key, value in pairs(l2) do
    l2[key] = torch.FloatTensor(value)
  end

  return mean, std, l2
end

function DataBinary:plotDataStatistics(mean, std, l2)
  local function PlotStats(container, nameStr)
    assert(gnuplot ~= nil, 'Plotting dataset stats requires gnuplot')
    local numKeys = 0
    for key, _ in pairs(container) do
      numKeys = numKeys + 1
    end
    local nrow = math.floor(math.sqrt(numKeys))
    local ncol = math.ceil(numKeys / nrow)
    gnuplot.figure()
    gnuplot.raw('set multiplot layout ' .. nrow .. ',' .. ncol)
    local i = 0
    for key, value in pairs(container) do
      gnuplot.raw("set title '" .. self.prefix .. ' ' .. nameStr .. ': ' ..
                  key .. "'")
      gnuplot.raw('set xtics font ", 8" rotate by 45 right')
      gnuplot.raw('set ytics font ", 8"')
      gnuplot.hist(value, 100)
    end
    gnuplot.raw('unset multiplot')
  end
  PlotStats(mean, 'mean')
  PlotStats(std, 'std')
  PlotStats(l2, 'l2')
end
