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
local tfluids = torch.loadPackageSafe('tfluids')
if tfluids == nil then
  tfluids = {}
  torch.makeGlobal('tfluids', tfluids)
end
local torchzlib = torch.loadPackageSafe('torchzlib')
local torchzfp = torch.loadPackageSafe('torchzfp')
local paths = require('paths')

local DataBinary, parent = torch.class('torch.DataBinary')

local COMPRESS_DATA_ON_DISK = false
local ZFP_ACCURACY = 1e-5
local DIV_THRESHOLD = 100  -- Relaxed threshold.
local DEFAULT_BORDER_WIDTH = 1

-- Helper function to load in and perform assertions on a run timestep file.
-- This is a little naughty. Dump tfluids in the tfluids namespace so it can
-- be used externally.
-- @param bWidth: How much of the border to remove.
function tfluids.loadFile(fn, bWidth)
  bWidth = bWidth or DEFAULT_BORDER_WIDTH
  assert(paths.filep(fn), "couldn't find ".. fn)
  local file = torch.DiskFile(fn, 'r')
  file:binary()
  local transpose = file:readInt()
  local nx = file:readInt()
  local ny = file:readInt()
  local nz = file:readInt()
  local time = file:readFloat()
  local numel = nx * ny * nz
  local Ux = torch.FloatTensor(file:readFloat(numel))
  local Uy = torch.FloatTensor(file:readFloat(numel))
  local Uz = torch.FloatTensor(file:readFloat(numel))
  local p = torch.FloatTensor(file:readFloat(numel))
  local geom = torch.FloatTensor(file:readFloat(numel))
  local minVal = 0.0

  Ux:resize(nz, ny, nx)
  Uy:resize(nz, ny, nx)
  Uz:resize(nz, ny, nx)
  p:resize(nz, ny, nx)
  geom:resize(nz, ny, nx)

  -- Ignore the border pixels.
  local xrange = {bWidth + 1, nx - bWidth}
  local yrange = {bWidth + 1, ny - bWidth}
  local zrange = {}
  if nz > 1 then
    zrange = {bWidth + 1, nz - bWidth}
  end
  p = p[{zrange, yrange, xrange}]:contiguous()
  Ux = Ux[{zrange, yrange, xrange}]:contiguous()
  Uy = Uy[{zrange, yrange, xrange}]:contiguous()
  Uz = Uz[{zrange, yrange, xrange}]:contiguous()
  geom = geom[{zrange, yrange, xrange}]:contiguous()
  if nz == 1 then
    minVal = math.min(torch.min(p), torch.min(Ux), torch.min(Uy))
  else
    minVal = math.min(torch.min(p), torch.min(Ux), torch.min(Uy), torch.min(Uz))
  end

  return time, p, Ux, Uy, Uz, geom,  minVal
end

function DataBinary:__init(conf, prefix)
  -- Load the processed data.
  self.dataDir = conf.dataDir  -- Just for reference
  self.dataset = conf.dataset  -- Just for reference
  self.prefix = prefix
  local baseDir = self:_getBaseDir(conf) .. '/'
  local runDirs = torch.ls(baseDir)
  assert(#runDirs > 0, "couldn't find any run directories in " .. baseDir)

  -- Create a network to calculate divergence of the set (see comment below).
  local divNet = nn.VelocityDivergence()

  -- Since the flat files array will include runs from multiple simulations, we
  -- need a list of valid frame pairs (to avoid trying to predict the transition
  -- between the last frame in one run and the first in another).
  self.runs = {}  -- Not necessarily the entire set (we may remove some).
  self.minValue = math.huge
  for i = 1, #runDirs do
    torch.progress(i, #runDirs)
    local runDir = self:_getBaseDir(conf) .. '/' .. runDirs[i] .. '/'
    local files = torch.ls(runDir .. '*[0-9].bin')
    local divFiles = torch.ls(runDir .. '*[0-9]_divergent.bin')
    
    assert(#files == #divFiles,
           "Main file count not equal to divergence file count")
    assert(#files > conf.ignoreFrames, 'Not enough files in sub-dir ' .. runDir)

    -- Ignore the first n frames.
    local runFiles = {}
    local runFilesDivergence = {}
    for f = conf.ignoreFrames + 1, #files do
      runFiles[#runFiles + 1] = files[f]
      runFilesDivergence[#runFilesDivergence + 1] = divFiles[f]
    end

    local data = {}
    local dataMin = math.huge;
    for f = 1, #runFiles do
      collectgarbage()
      local time, p, Ux, Uy, Uz, geom, minVal = tfluids.loadFile(runFiles[f])
      local _, pDiv, UxDiv, UyDiv, UzDiv, _, _ =
          tfluids.loadFile(runFilesDivergence[f])

      dataMin = math.min(minVal, dataMin)

      if data.p == nil then
        -- Initialize the data structure.
        local xdim = p:size(3)
        local ydim = p:size(2)
        local zdim = p:size(1)
        data.p = torch.FloatTensor(#runFiles, zdim, ydim, xdim)
        data.geom = torch.FloatTensor(#runFiles, zdim, ydim, xdim)
        data.Ux = torch.FloatTensor(#runFiles, zdim, ydim, xdim)
        data.Uy = torch.FloatTensor(#runFiles, zdim, ydim, xdim)
        data.Uz = torch.FloatTensor(#runFiles, zdim, ydim, xdim)
        data.pDiv = torch.FloatTensor(#runFiles, zdim, ydim, xdim)
        data.UxDiv = torch.FloatTensor(#runFilesDivergence, zdim, ydim, xdim)
        data.UyDiv = torch.FloatTensor(#runFilesDivergence, zdim, ydim, xdim)
        data.UzDiv = torch.FloatTensor(#runFilesDivergence, zdim, ydim, xdim)
        data.time = torch.FloatTensor(#runFiles)
      end

      -- We could be pedantic here and start checking that the sizes of all
      -- files match, but instead just let the copy fail if numel doesn't match
      -- and assume the dims do.
      data.p[f]:copy(p)
      data.geom[f]:copy(geom)
      data.Ux[f]:copy(Ux)
      data.Uy[f]:copy(Uy)
      data.Uz[f]:copy(Uz)
      data.pDiv[f]:copy(pDiv)
      data.UxDiv[f]:copy(UxDiv)
      data.UyDiv[f]:copy(UyDiv)
      data.UzDiv[f]:copy(UzDiv)
      data.time[f] = time
    end

    local xdim = data.p:size(4)
    local ydim = data.p:size(3)
    local zdim = data.p:size(2)

    -- Unfortunately, some samples might be the result of an unstable Manta sim.
    -- i.e. the divergence blew up, but not completely. If this is the
    -- case, then we should ignore the entire run. So we need to check the
    -- divergence of the target velocity.
    -- NOTE: the target velocity is the result of a MAC grid --> central
    -- conversion. So actually, the target velocity will have SOME divergence
    -- due to interpolation error and differences between Manta and our
    -- divergence calculation.
    local sz = {#runFiles, 1, zdim, ydim, xdim}
    local UTarget = torch.cat({data.Ux:view(unpack(sz)),
                               data.Uy:view(unpack(sz)),
                               data.Uz:view(unpack(sz))}, 2)
    local divTarget = divNet:forward({UTarget, data.geom})
    local maxDiv = divTarget:max()
    if maxDiv > DIV_THRESHOLD then
      print('WARNING: run ' .. runDirs[i] .. ' (i = ' .. i ..
            ') has a sample with max(div) = ' .. maxDiv ..
            ' which is above the allowable threshold (' .. DIV_THRESHOLD .. ')')
      print('  --> Removing run from the dataset...')
    else
      -- This sample is OK.
      self.minValue = math.min(dataMin, self.minValue)

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
        zdim = zdim
      }
    end
  end
  runDirs = nil  -- Full set no longer needed.

  -- Check that the size of each run is the same.
  self.xdim = self.runs[1].xdim
  self.ydim = self.runs[1].ydim
  self.zdim = self.runs[1].zdim
  for i = 1, #self.runs do
    assert(self.xdim == self.runs[i].xdim)
    assert(self.ydim == self.runs[i].ydim)
    assert(self.zdim == self.runs[i].zdim)
  end

  self.twoDim = self.zdim == 1

  -- Create an explicit list of trainable frames.
  self.samples = {}
  for r = 1, #self.runs do
    for f = 1, self.runs[r].ntimesteps  do
      self.samples[#self.samples + 1] = {r, f}
    end
  end
  self.samples = torch.LongTensor(self.samples)
end

function DataBinary:_getBaseDir(conf)
  -- Sanity check the dataset name (note: the dataDir might have changed).
  assert(conf.dataset == self.dataset,
         'trying to get data from the wrong dataset')
  return conf.dataDir .. '/' .. conf.dataset .. '/' .. self.prefix .. '/'
end

-- Note: runDir is typically self.runs[irun].dir
function DataBinary:_getCachePath(conf, runDir, iframe)
  local dir = self:_getBaseDir(conf) .. '/' .. runDir .. '/'
  local fn = string.format('frame_cache_%06d', iframe)
  return dir .. fn
end

function DataBinary:_cacheDataToDisk(data, conf, runDir)
  -- The function will take in a table of tensors and an output directory
  -- and store the result on disk.

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
    torch.save(self:_getCachePath(conf, runDir, iframe), frameData)
  end
end

function DataBinary:_loadDiskCache(conf, irun, iframe)
  assert(irun <= #self.runs and irun >= 1)
  assert(iframe <= self.runs[irun].ntimesteps and iframe >= 1)
  local fn = self:_getCachePath(conf, self.runs[irun].dir, iframe)
  local data = torch.load(fn)
  for key, value in pairs(data) do
    if torch.type(value) == 'torch.ZFPTensor' then
      data[key] = value:decompress()
    end
  end
  return data
end

-- Abstract away getting data from however we store it.
function DataBinary:getSample(conf, irun, iframe)
  assert(irun <= #self.runs and irun >= 1, 'getSample: irun out of bounds!')
  assert(iframe <= self.runs[irun].ntimesteps and iframe >= 1,
    'getSample: iframe out of bounds!')

  local data = self:_loadDiskCache(conf, irun, iframe)

  local p = data.p
  local pDiv = data.pDiv
  local geom = data.geom
  local Ux = data.Ux
  local Uy = data.Uy
  local UxDiv = data.UxDiv
  local UyDiv = data.UyDiv
  local Uz, UzDiv
  if not self.twoDim then
    Uz = data.Uz
    UzDiv = data.UzDiv
  end
  
  return p, Ux, Uy, Uz, geom, pDiv, UxDiv, UyDiv, UzDiv
end

function DataBinary:saveSampleToMatlab(conf, filename, irun, iframe)
  local p, Ux, Uy, Uz, geom, pDiv, UxDiv, UyDiv, UzDiv =
      self:getSample(conf, irun, iframe)
  local data = {p = p, Ux = Ux, Uy = Uy, Uz = Uz, geom = geom,
                pDiv = pDiv, UxDiv = UxDiv, UyDiv = UyDiv, UzDiv = UzDiv}
  for key, value in pairs(data) do
    data[key] = value:double():squeeze()
  end
  assert(mattorch ~= nil, 'saveSampleToMatlab requires mattorch.')
  mattorch.save(filename, data)
end

-- @param, irun: index of the simulation run you want to visualize.
-- @param, depth: <OPTIONAL> for 3D datasets, this is the z slice to visualize.
function DataBinary:visualizeData(conf, irun, depth)
  if irun ~= nil then
    assert(irun >= 1 and irun <= #self.runs)
  else
    irun = torch.random(1, #self.runs)
  end
  print('Run: ' .. irun .. ', dir = ' .. self.runs[irun].dir)
  depth = depth or 1

  local ntimesteps = self.runs[irun].ntimesteps
  local nrow = math.ceil(math.sqrt(ntimesteps))
  local zoom = 1024 / (self.xdim * nrow)
  zoom = math.min(zoom, 4)

  local data = {}
  data.geom = torch.FloatTensor(ntimesteps, 1, self.zdim, self.ydim,
                                self.xdim)
  data.p = torch.FloatTensor(ntimesteps, 1, self.zdim, self.ydim, self.xdim)
  if self.twoDim then
    data.U = torch.FloatTensor(ntimesteps, 2, self.zdim, self.ydim, self.xdim)
  else
    data.U = torch.FloatTensor(ntimesteps, 3, self.zdim, self.ydim, self.xdim)
  end

  for i = 1, ntimesteps do
    local p, Ux, Uy, Uz, geom, pDiv, UxDiv, UyDiv, UzDiv =
        self:getSample(conf, irun, i)
    data.geom[i]:copy(geom)
    data.p[i]:copy(p)
    data.U[i][1]:copy(Ux)
    data.U[i][2]:copy(Uy)
    if not self.twoDim then
      data.U[i][3]:copy(Uz)
    end
  end

  self:_visualizeBatchData(data, self.runs[irun].dir, depth)

  return data
end

-- Visualize the components of a concatenated tensor of (p, U, geom).
function DataBinary:_visualizeBatchData(data, legend, depth)
  -- Check the input.
  assert(data.p ~= nil and data.geom ~= nil and data.U ~= nil)

  legend = legend or ''
  depth = depth or 1
  local nrow = math.floor(math.sqrt(data.geom:size(1)))
  local zoom = 1024 / (nrow * self.xdim)
  zoom = math.min(zoom, 2)
  local twoDim = data.U:size(2) == 2

  local p, Ux, Uy, geom, Uz

  -- Just visualize one depth slice.
  p = data.p:float():select(3, depth)
  Ux = data.U[{{}, {1}}]:float():select(3, depth)
  Uy = data.U[{{}, {2}}]:float():select(3, depth)
  geom = data.geom:float():select(3, depth)

  local scaleeach = false;
  image.display{image = p, zoom = zoom, padding = 2, nrow = nrow,
    legend = (legend .. ': p'), scaleeach = scaleeach}
  if Ux ~= nil then
  image.display{image = Ux, zoom = zoom, padding = 2, nrow = nrow,
    legend = (legend .. ': Ux'), scaleeach = scaleeach}
  end
  if Uy ~= nil then
  image.display{image = Uy, zoom = zoom, padding = 2, nrow = nrow,
    legend = (legend .. ': Uy'), scaleeach = scaleeach}
  end
  if geom ~= nil then
    image.display{image = geom, zoom = zoom, padding = 2, nrow = nrow,
      legend = (legend .. ': geom'), scaleeach = scaleeach}
  end
  if Uz ~= nil then
    image.display{image = Uz, zoom = zoom, padding = 2, nrow = nrow,
      legend = (legend .. ': Uz'), scaleeach = scaleeach}
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
  depth = depth or 1

  local batchCPU, batchGPU = self:AllocateBatchMemory(conf, mconf)
  local perturb = conf.trainPerturb.on
  local degRot, scale, transVPix, transUPix, flipX, flipY, flipZ =
    self:CreateBatch(batchCPU, batchGPU, conf, mconf, imgList, perturb)

  print('    Image set:')  -- Don't print if this is used in the train loop.
  for i = 1, #imgList do
    local isample = imgList[i]

    local irun = self.samples[isample][1]
    local if1 = self.samples[isample][2]
    
    print(string.format("    %d - sample %d: irun = %d, if1 = %d", i,
                        imgList[i], irun, if1))
    print(string.format(
        "      degRot = %.1f, scale = %.3f, trans_uv = [%.1f, %.1f], " ..
        "flip = (%d, %d, %d)", degRot[i], scale[i], transUPix[i],
        transVPix[i], flipX[i], flipY[i], flipZ[i]))
  end

  local range = {1, #imgList}
  for key, value in pairs(batchGPU) do
    batchGPU[key] = value[{range}]  -- Just pick the samples in the imgList.
  end

  local inputData = {
      p = batchGPU.pDiv,
      U = batchGPU.UDiv,
      geom = batchGPU.geom
  }
  self:_visualizeBatchData(inputData, 'input', depth)

  local outputData = {
      p = batchGPU.pTarget,
      U = batchGPU.UTarget,
      geom = batchGPU.geom
  }
  self:_visualizeBatchData(outputData, 'target', depth)

  return batchCPU, batchGPU
end

function DataBinary:AllocateBatchMemory(conf, mconf)
  -- Create the data containers.
  local d = self.zdim
  local h = self.ydim
  local w = self.xdim

  assert(self.twoDim == mconf.twoDim,
    'Mixing 3D model with 2D data or vice-versa!')

  local batchCPU = {}

  local numUChans = 3
  if self.twoDim then
    numUChans = 2
  end
  batchCPU.pDiv = torch.FloatTensor(conf.batchSize, 1, d, h, w):normal()
  batchCPU.pTarget = batchCPU.pDiv:clone()
  batchCPU.UDiv = torch.FloatTensor(conf.batchSize, numUChans, d, h, w):normal()
  batchCPU.UTarget = batchCPU.UDiv:clone()
  batchCPU.geom = torch.FloatTensor(conf.batchSize, 1, d, h, w):normal()

  local batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')

  return batchCPU, batchGPU
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
function DataBinary:CreateBatch(batchCPU, batchGPU, conf, mconf, imgList,
                                perturb)
  assert(mconf.netDownsample == 1, "pooling not supported")
  if perturb == nil then
    perturb = false
  end

  local d = self.zdim
  local h = self.ydim
  local w = self.xdim
  local numUChans = 3
  if self.twoDim then
    numUChans = 2
  end

  -- Make sure we haven't done anything silly.
  assert(self.twoDim == mconf.twoDim,
         'Mixing 3D model with 2D data or vice-versa!')

  -- I used to pre-allocate these, but the code is much more redable if we just
  -- allocate them every time CreateBatch is called
  local degRot = torch.FloatTensor(conf.batchSize)
  local scale = torch.FloatTensor(conf.batchSize)
  local transUPix = torch.FloatTensor(conf.batchSize)
  local transVPix = torch.FloatTensor(conf.batchSize)
  local flipX = torch.ByteTensor(conf.batchSize)
  local flipY = torch.ByteTensor(conf.batchSize)
  local flipZ = torch.ByteTensor(conf.batchSize)
  local pDiv, geom, UDiv

  pDiv = torch.FloatTensor(1, self.zdim, self.ydim, self.xdim)
  geom = torch.FloatTensor(1, self.zdim, self.ydim, self.xdim)
  UDiv = torch.FloatTensor(numUChans, self.zdim, self.ydim, self.xdim)
  local pTarget = pDiv:clone()
  local UTarget = UDiv:clone()

  degRot:fill(0)
  scale:fill(1)
  transUPix:fill(0)
  transVPix:fill(0)
  flipX:fill(0)
  flipY:fill(0)
  flipZ:fill(0)

  -- create mini batch
  for i = 1, #imgList do
    -- For each image in the batch list
    local isample = imgList[i]
    assert(isample >= 1 and isample <= self:nsamples())
    local irun = self.samples[isample][1]
    local if1 = self.samples[isample][2]

    -- Randomly perturb the data
    if perturb then
      -- Sample perturbation parameters.
      assert(conf.trainPerturb.scale == 0, 'Scale not yet supported.')
      scale[i] = 0
      assert(conf.trainPerturb.transPix == 0, 'Translation not yet supported.')
      transVPix[i] = 0
      transUPix[i] = 0
      assert(conf.trainPerturb.rotation == 0, 'Rotation not yet supported.')
      degRot[i] = 0
      flipX[i] = torch.bernoulli(conf.trainPerturb.flipProb)
      flipY[i] = torch.bernoulli(conf.trainPerturb.flipProb)
      if not self.twoDim then
        flipZ[i] = torch.bernoulli(conf.trainPerturb.flipProb)
      else
        flipZ[i] = 0
      end
    end

    -- Copy the input channels
    local p1, Ux1, Uy1, Uz1, geom1, p1Div, Ux1Div, Uy1Div, Uz1Div =
        self:getSample(conf, irun, if1)
    pDiv:copy(p1Div)
    geom:copy(geom1)
    UDiv[1]:copy(Ux1Div)
    UDiv[2]:copy(Uy1Div)
    if not self.twoDim then
      assert(numUChans == 3)
      UDiv[3]:copy(Uz1Div)
    end

    local function flip(tensor, dim)
      -- Define global temporary buffer so that we don't needlessly malloc.
      if torch._flipTensor == nil then
        torch._flipTensor = torch.FloatTensor()
      end
      torch._flipTensor:resizeAs(tensor)
      image.flip(torch._flipTensor, tensor, dim)
      tensor:copy(torch._flipTensor)
    end

    local function performFlips(tensor, flipX, flipY, flipZ, UField)
      if flipX == 1 then
        flip(tensor, 4)  -- Flip horiz dim 4.
      end
      if flipY == 1 then
        flip(tensor, 3)  -- Flip vert dim 3.
      end
      if flipZ == 1 then
        assert(not self.twoDim)
        flip(tensor, 2)  -- Flip depth dim 2.
      end
      if UField then
        -- Since this is a vector field, we'll also need to flip the
        -- vector itself (in addition to flipping the tensor elements).
        if flipX == 1 then
          tensor[1]:mul(-1)
        end
        if flipY == 1 then
          tensor[2]:mul(-1)
        end
        if flipZ == 1 then
          assert(not self.twoDim)
          tensor[3]:mul(-1)
        end
      end
    end

    -- Perturb the input channels.
    if perturb then
      performFlips(pDiv, flipX[i], flipY[i], flipZ[i], false)
      performFlips(UDiv, flipX[i], flipY[i], flipZ[i], true)
      performFlips(geom, flipX[i], flipY[i], flipZ[i], false)
    end

    -- Copy the input channels to the batch array.
    batchCPU.pDiv[i]:copy(pDiv)
    batchCPU.geom[i]:copy(geom)
    batchCPU.UDiv[i]:copy(UDiv)

    -- Target is the corrected velocity field.
    pTarget:copy(p1)
    UTarget[1]:copy(Ux1)
    UTarget[2]:copy(Uy1)
    if not self.twoDim then
      UTarget[3]:copy(Uz1)
    end

    -- Perturb the output channels.
    if perturb then
      performFlips(pTarget, flipX[i], flipY[i], flipZ[i], false)
      performFlips(UTarget, flipX[i], flipY[i], flipZ[i], true)
    end

    -- Copy the output channels to the batch array.
    batchCPU.pTarget[i]:copy(pTarget)
    batchCPU.UTarget[i]:copy(UTarget)
  end

  -- Copy the data to the GPU in one (asynchronous) copy
  if batchGPU ~= nil then
    for key, value in pairs(batchCPU) do
      assert(batchGPU[key] ~= nil)
      batchGPU[key]:copy(value)
    end
  end

  return degRot, scale, transVPix, transUPix, flipX, flipY, flipZ
end

function DataBinary:nsamples()
  -- Return the maximum number of image pairs we can support
  return self.samples:size(1)
end

