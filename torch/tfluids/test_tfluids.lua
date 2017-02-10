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

-- Adhoc testing for our utility functions.
--
-- Note: these tests MUST be run from the FluidNet/torch/tfluids directory!

-- You can easily test specific units like this:
-- qlua -ltfluids -lenv -e "tfluids.test{'velocityDivergence'}"

-- Or to test everything:
-- qlua -ltfluids -lenv -e "tfluids.test()"

local nn = require('nn')
local paths = require('paths')
local sys = require('sys')

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-6
local precisionCPUGPU = 1e-5  -- All in floats.
local loosePrecision = 1e-4
local mytester = torch.Tester()
local test = torch.TestSuite()
local times = {}
local profileTimeSec = 1  -- Probably too small for any real profiling...
local profileResolution = 128
local jac = nn.Jacobian

local function tileToResolution(x, res)
  assert(torch.isTensor(x) and x:dim() == 5)
  local is3D = x:size(3) > 1
  local ntile = {}
  for i = 1, 3 do
    ntile[i] = math.ceil(res / x:size(2 + i))
    assert(ntile[i] > 1, 'tileToResolution can only upscale')
  end
  if not is3D then
    ntile[1] = 1
  end
  x = x:repeatTensor(1, 1, ntile[1], ntile[2], ntile[3]):contiguous()
  if not is3D then
    x = x[{{}, {}, {}, {1, res}, {1, res}}]:contiguous()
  else
    x = x[{{}, {}, {1, res}, {1, res}, {1, res}}]:contiguous()
  end
  return x
end

-- Result args is an optional array of argument indices that are result
-- tensors, i.e. the output of the function is stored in these tensors.
-- We will fill these tensors with random to make sure that the GPU and
-- CPU versions MUST overwrite them to be correct. Use 'nil' if all the
-- arguments are inputs.
local function profileAndTestCuda(func, name, args, resultArgs,
                                  resizeForProfile)
  if not tfluids.withCUDA then
    print('WARNING: tfluids compiled without cuda. Not testing.')
    return
  end
  local cutorch = require('cutorch')
  if resizeForProfile == nil then
    resizeForProfile = true
  end
  local tm = {}
  times[name] = tm

  -- Make a copy of the input arguments on the GPU and CPU (as float)
  local argsInGPU = {}
  local argsInCPU = {}
  for key, value in pairs(args) do
    if torch.isTensor(value) then
      if value:type() == 'torch.CudaTensor' then
        argsInGPU[key] = value:clone()
      else
        argsInGPU[key] = value:cuda()  -- Implicit clone.
      end
      if value:type() == 'torch.FloatTensor' then
        argsInCPU[key] = value:clone()
      else
        argsInCPU[key] = value:float()  -- Implicit clone.
      end
    else
      argsInGPU[key] = value
      argsInCPU[key] = value
    end
  end

  if resultArgs ~= nil then
    for _, key in pairs(resultArgs) do
      argsInCPU[key]:uniform()
      argsInGPU[key]:uniform()
    end
  end

  args = nil  -- We no longer need it.

  -- Now call the CPU function.
  local resCPU = {func(unpack(argsInCPU))}  -- Capture multiple return args.

  -- Now call the GPU function.
  local resGPU = {func(unpack(argsInGPU))}

  assert(#resCPU == #resGPU)

  -- Make sure all the input args are now the same (if modified in place).
  for key, value in pairs(argsInCPU) do
    if torch.isTensor(value) then
      local err = argsInCPU[key] - argsInGPU[key]:float()
      mytester:assertlt(err:abs():max(), precisionCPUGPU,
                        'Error CPU:GPU mismatch: ' .. name .. ', arg # ' .. key)
    else
      -- This is pedantic, but do it anyway.
      mytester:assert(argsInCPU[key] == argsInGPU[key])
    end
  end

  -- Make sure the return value is the same. Note: this will break 
  for key, value in pairs(resCPU) do
    if torch.isTensor(value) then
      local err = resCPU[key] - resGPU[key]:float()
      mytester:assertlt(err:abs():max(), precisionCPUGPU,
                        'Error CPU:GPU mismatch: ' .. name)
    else
      -- This is pedantic, but do it anyway.
      mytester:assert(resCPU[key] == resGPU[key])
    end
  end

  if resizeForProfile then
    -- Now resize the arrays to something more reasonable for profiling.
    for key, value in pairs(argsInCPU) do
      if torch.isTensor(value) and value:dim() == 5 then
        -- Upscale by tiling. Technically we don't really care what the values
        -- are when profiling, but this will at least keep them from being NaNs
        -- and with reasonable flag values.
        argsInCPU[key] = tileToResolution(argsInCPU[key], profileResolution)
        argsInGPU[key] = tileToResolution(argsInGPU[key], profileResolution)
      end
    end
  end

  -- Profile CPU.
  local count = 0
  sys.tic()
  while sys.toc() < profileTimeSec do
    count = count + 1
    func(unpack(argsInCPU))
  end
  tm.cpu = sys.toc() / count

  -- Profile GPU.
  count = 0
  sys.tic()
  while sys.toc() < profileTimeSec do
    count = count + 1
    func(unpack(argsInGPU))
  end
  cutorch.synchronize()
  tm.gpu = sys.toc() / count
end

local function assertNotAllEqual(tensor)
  -- Make sure we load samples from file that aren't all the same, otherwise
  -- we're not really testing the batch dimension.
  if tensor:size(1) == 1 then
    return  -- Only one sample, so it's correct.
  end
  local first = tensor[{{1}}]
  local others = tensor[{{2, tensor:size(1)}}]
  assert((others - first:expandAs(others)):abs():max() > 1e-5,
         'All samples equal!')
end

-- A shorthand to call torch.loadMantaFile on multiple files to concat them
-- into a single batch.
local function loadMantaBatch(fn)
  local files = torch.ls('test_data/b*_' .. fn)
  assert(#files == 16, 'Hard-coded just in case something stupid happens')
  
  local p = {}
  local U = {}
  local flags = {}
  local density = {}
  local is3D = nil
  for _, file in pairs(files) do
    local curP, curU, curFlags, curDensity, curIs3D = torch.loadMantaFile(file)
    p[#p + 1] = curP
    U[#U + 1] = curU
    flags[#flags + 1] = curFlags
    density[#density + 1] = curDensity
    if is3D == nil then
      is3D = curIs3D
    else
      assert(is3D == curIs3D)
    end
  end
  p = torch.cat(p, 1)
  U = torch.cat(U, 1)
  flags = torch.cat(flags, 1)
  density = torch.cat(density, 1)

  local function assertNotAllEqual(tensor)
    -- Make sure we load samples from file that aren't all the same, otherwise
    -- we're not really testing the batch dimension.
    assert(tensor:size(1) > 1, 'only one sample')
    local first = tensor[{{1}}]
    local others = tensor[{{2, tensor:size(1)}}]
    assert((others - first:expandAs(others)):abs():max() > 1e-5,
           'All samples equal!')
  end

  return p, U, flags, density, is3D
end

function test.advectManta()
  for dim = 2, 3 do
    -- Load the pre-advection Manta file for this test.
    local fn = dim .. 'd_initial.bin'
    local _, U, flags, density, is3D = loadMantaBatch(fn)
    assertNotAllEqual(U)
    assertNotAllEqual(flags)
    assertNotAllEqual(density)

    assert(is3D == (dim == 3))

    -- Now do advection using the 2 parameters and check against Manta.
    for _, method in pairs({'euler', 'maccormack'}) do
      -- Load the Manta ground truth.
      local order
      if method == 'euler' then
        order = 1
      else
        order = 2
      end
      local openStr = 'False'  -- It actually doesn't affect advection.
      fn = (dim .. 'd_advect_openBounds_' .. openStr .. '_order_' .. order ..
            '.bin')
      local _, UManta, flagsDiv, densityManta, is3D = loadMantaBatch(fn)
      assert(is3D == (dim == 3))
      assert(torch.all(torch.eq(flags, flagsDiv)), 'flags changed!')

      -- Make sure that Manta didn't change the flags.
      assert((flags - flagsDiv):abs():max() == 0, 'Flags changed!')

      -- Perform our own advection.
      local dt = 0.1  -- Unfortunately hard coded for now.
      local boundaryWidth = 0  -- Also shouldn't be hard coded.
      local maccormackStrength = 1.0

      local nameS = ('tfluids.advectScalar dim ' .. dim .. ', method ' ..
                     method)
      local nameU = ('tfluids.advectVel dim ' .. dim .. ', method ' .. method)

      -- Note: the clone's here are to make sure every inner loops
      -- sees completely independent data.
      local densityAdv =
          torch.rand(unpack(density:size():totable())):typeAs(density)
      tfluids.advectScalar(dt, density:clone(), U:clone(), flags:clone(),
                           method, densityAdv, nil, maccormackStrength,
                           boundaryWidth)
      local err = densityManta - densityAdv
      mytester:assertlt(err:abs():max(), precision, 'Error: ' .. nameS)

      -- Also try an in-place scalar advection.
      densityAdv = density:clone()
      tfluids.advectScalar(dt, densityAdv, U:clone(), flags:clone(),
                           method, nil, nil, maccormackStrength, boundaryWidth)
      local err = densityManta - densityAdv
      mytester:assertlt(err:abs():max(), precision, 'Error: ' .. nameS)

      local UAdv =
          torch.rand(unpack(U:size():totable())):typeAs(U)
      tfluids.advectVel(dt, U:clone(), flags:clone(), method, UAdv,
                        maccormackStrength, boundaryWidth)
      err = UManta - UAdv
      mytester:assertlt(err:abs():max(), precision, 'Error: ' .. nameU)

      -- Also try an in-place velocity advection.
      local UAdv = U:clone()
      tfluids.advectVel(dt, UAdv, flags:clone(), method, nil, 
                        maccormackStrength, boundaryWidth)
      err = UManta - UAdv
      mytester:assertlt(err:abs():max(), precision, 'Error: ' .. nameU)

      -- Now test and profile the CUDA version.
      profileAndTestCuda(tfluids.advectScalar, nameS,
                         {dt, density, U, flags,
                          method, densityAdv, nil, maccormackStrength,
                          boundaryWidth}, {6})

      profileAndTestCuda(tfluids.advectVel, nameU,
                         {dt, U, flags, method, U:clone():fill(0),
                          maccormackStrength, boundaryWidth}, {5})
    end
  end  -- for dim
end

function test.advectOurs()
  for dim = 2, 3 do
    -- Now do advection using the 2 parameters and check against Manta.
    for _, method in pairs({'eulerOurs', 'maccormackOurs', 'rk2Ours',
                            'rk3Ours'}) do
      for sampleOutsideFluid = 0, 1 do
        local fn = dim .. 'd_initial.bin'
        local _, U, flags, density, is3D = loadMantaBatch(fn)
        assertNotAllEqual(U)
        assertNotAllEqual(flags)
        assertNotAllEqual(density)
        assert(is3D == (dim == 3))

        local dt = 0.1
        local boundaryWidth = 0
        local maccormackStrength = torch.uniform()  -- in [0, 1]

        local nameS = ('advectScalar dim ' .. dim .. ', method ' ..
                       method .. ', sampleOutFluid ' .. sampleOutsideFluid)
        local nameU = ('advectVel dim ' .. dim .. ', method ' ..
                       method .. ', sampleOutFluid ' .. sampleOutsideFluid)

        -- We don't have GT for the non-manta implementations. Just compare
        -- CPU and GPU implementations for consistency.
        local densityAdv = density:clone():uniform()
        profileAndTestCuda(tfluids.advectScalar, nameS,
                           {dt, density, U, flags, method, densityAdv,
                            sampleOutsideFluid == 1, maccormackStrength,
                            boundaryWidth}, {6})

        local UAdv = U:clone():uniform()
        profileAndTestCuda(tfluids.advectVel, nameU,
                           {dt, U, flags, method, UAdv,
                            maccormackStrength, boundaryWidth}, {5})
      end
    end
  end
end

local function createJacobianTestData(p, U, flags, density)
  assert(U ~= nil, 'At least U must be non-nil')
  local is3D = U:size(2) == 3
  local zStart = 1
  local zSize
  if is3D then
    -- In 3D a Jacobian of numel * numel is too big! Do a slice of the
    -- z dimension.
    zSize = 4
  else
    zSize = U:size(3)
  end
  -- Also do a slice of bsize.
  local bsize = math.min(U:size(1), 3)
  if p ~= nil then
    p = p:narrow(3, zStart, zSize):contiguous():double()
    p = p:narrow(1, 1, bsize):contiguous():double()
  end
  U = U:narrow(3, zStart, zSize):contiguous():double()
  U = U:narrow(1, 1, bsize):contiguous():double()
  if flags ~= nil then
    flags = flags:narrow(3, zStart, zSize):contiguous():double()
    flags = flags:narrow(1, 1, bsize):contiguous():double()
    -- Make sure the flags array is still interesting (i.e that there
    -- is obstacles).
    assert((flags - tfluids.CellType.TypeFluid):abs():max() > 0,
           'All cells are fluid.')
  end
  if density ~= nil then
    density = density:narrow(3, zStart, zSize):contiguous():double()
    density = density:narrow(1, 1, bsize):contiguous():double()
  end
  return p, U, flags, density
end

local function createJacobianTestModel(mod, flags)
  local testMod = nn.Sequential()
  -- InjectTensor is a specially designed cell that inputs a tensor and
  -- outputs a table of {tensor, flags}. It is so that the input output
  -- function looks like a tensor to tensor op (where flags are constant).
  testMod:add(nn.InjectTensor(flags))
  testMod:add(mod)
  return testMod
end

function test.setWallBcs()
  local function testSetWallBcs(dim, fnInput, fnOutput)
    local fn = dim .. 'd_' .. fnInput
    local _, U, flags, _, is3D = loadMantaBatch(fn)
    assertNotAllEqual(U)
    assertNotAllEqual(flags)

    assert(is3D == (dim == 3))

    -- Make sure they're not all fluid cells (i.e. that there is some obstacles
    -- in the sub-set) otherwise we're not testing anything.
    assert((flags - tfluids.CellType.TypeFluid):abs():max() > 0,
           'All cells are fluid!')

    fn = dim .. 'd_' .. fnOutput
    local _, UManta, flagsManta, _, is3D = loadMantaBatch(fn)
    assert(is3D == (dim == 3))
    assert(torch.all(torch.eq(flags, flagsManta)), 'flags changed!')

    -- Perform our own setWallBcs
    local UOurs = U:clone()
    tfluids.setWallBcsForward(UOurs, flags)
    local err = UOurs - UManta
    mytester:assertlt(err:abs():max(), precision,
                      'Error: tfluids.setWallBcs dim ' .. dim)

    -- Test the same forward function but in the module.
    local mod = createJacobianTestModel(tfluids.SetWallBcs(), flags):float()
    err = mod:forward(U) - UManta
    mytester:assertlt(err:abs():max(), precision,
                     'Error: tfluids.setWallBcs FPROP dim ' .. dim)

    local _, UBprop, flagsBprop, _ = createJacobianTestData(nil, U, flags, nil)
    local mod = createJacobianTestModel(tfluids.SetWallBcs(), flagsBprop)
    err = jac.testJacobian(mod, UBprop)
    mytester:assertlt(math.abs(err), precision,
                      'Error: tfluids.setWallBcs BPROP dim ' .. dim)

    -- Now test and profile the CUDA version.
    profileAndTestCuda(tfluids.setWallBcsForward,
                       'setWallBcsForward_' .. dim .. 'd',
                       {U, flags})
  end

  for dim = 2, 3 do
    -- We call setWallBcs three times in the Manta training code so we should
    -- test it in all cases.
    testSetWallBcs(dim, 'advect.bin', 'setWallBcs1.bin')
    testSetWallBcs(dim, 'vorticityConfinement.bin', 'setWallBcs2.bin')
    testSetWallBcs(dim, 'solvePressure.bin', 'setWallBcs3.bin')
  end
end

function test.velocityDivergence()
  for dim = 2, 3 do
    -- Load the input Manta data.
    local fn = dim .. 'd_vorticityConfinement.bin'
    local _, U, flags, _, is3D = loadMantaBatch(fn)
    assertNotAllEqual(U)
    assertNotAllEqual(flags)

    assert(is3D == (dim == 3))
    -- Now load the output Manta data. Note: in manta/scenes/_testData.py the
    -- rhs (i.e. divergence) is stored in the pressure channel of our binary
    -- format.
    fn = dim .. 'd_makeRhs.bin'
    local divManta, UManta, flagsManta, _, is3D = loadMantaBatch(fn)
    assert(is3D == (dim == 3))
    assert(torch.all(torch.eq(U, UManta)), 'Velocity changed!')
    assert(torch.all(torch.eq(flags, flagsManta)), 'flags changed!')

    -- Perform our own divergence calculation.
    local divOurs =
        torch.rand(unpack(divManta:size():totable())):typeAs(divManta)
    tfluids.velocityDivergenceForward(U:clone(), flags:clone(), divOurs)
    local err = divManta - divOurs
    mytester:assertlt(err:abs():max(), precision,
                      ('Error: tfluids.velocityDivergenceForward dim ' ..
                       dim))

    -- Test the same forward function but in the module.
    local mod = createJacobianTestModel(tfluids.VelocityDivergence(),
                                        flags):float()
    err = mod:forward(U) - divManta
    mytester:assertlt(err:abs():max(), precision,
                     'Error: tfluids.velocityDivergence FPROP dim ' .. dim)

    local _, UBprop, flagsBprop, _ = createJacobianTestData(nil, U, flags, nil)
    local mod = createJacobianTestModel(tfluids.VelocityDivergence(),
                                        flagsBprop)
    err = jac.testJacobian(mod, UBprop)
    mytester:assertlt(math.abs(err), precision,
                      'Error: tfluids.velocityDivergence BPROP dim ' .. dim)

    -- Now test and profile the CUDA version.
    profileAndTestCuda(tfluids.velocityDivergenceForward,
                       'velocityDivergenceForward_' .. dim .. 'd',
                       {U, flags, divOurs}, {3})
    local gradOutput = divOurs:clone():uniform(0, 1)
    local gradU = U:clone()
    profileAndTestCuda(tfluids.velocityDivergenceBackward,
                       'velocityDivergenceBackward_' .. dim .. 'd',
                       {U, flags, gradOutput, gradU}, {4})
  end
end

function test.velocityUpdate()
  for dim = 2, 3 do
    -- Load the input Manta data.
    local fn = dim .. 'd_vorticityConfinement.bin'
    local _, U, flags, _, is3D = loadMantaBatch(fn)
    assertNotAllEqual(U)
    assertNotAllEqual(flags)

    assert(is3D == (dim == 3))
    -- Now load the output Manta data.
    fn = dim .. 'd_correctVelocity.bin'
    local pressure, UManta, flagsManta, _, is3D = loadMantaBatch(fn)
    assert(is3D == (dim == 3))
    assert(torch.all(torch.eq(flags, flagsManta)), 'flags changed!')

    -- Make sure this isn't a trivial velocity update (i.e. that velocities
    -- actually changed).
    assert((U - UManta):abs():max() > 1e-5, 'No velocities changed in Manta!')

    -- Perform our own velocity update calculation.
    local UOurs = U:clone()  -- This is the divergent U.
    tfluids.velocityUpdateForward(UOurs, flags:clone(), pressure:clone())
    local err = UManta - UOurs
    mytester:assertlt(err:abs():max(), precision,
                      ('Error: tfluids.velocityUpdateForward dim ' ..
                       dim))

    -- Test the same forward function but in the module.
    local mod = createJacobianTestModel(tfluids.VelocityUpdate(),
                                        {U, flags}):float()
    err = mod:forward(pressure) - UManta
    mytester:assertlt(err:abs():max(), precision,
                     'Error: tfluids.velocityUpdate FPROP dim ' .. dim)

    local pressureBprop, UBprop, flagsBprop, _ =
        createJacobianTestData(pressure, U, flags, nil)
    local mod = createJacobianTestModel(tfluids.VelocityUpdate(),
                                        {UBprop, flagsBprop})
    err = jac.testJacobian(mod, pressureBprop)
    mytester:assertlt(math.abs(err), precision,
                      'Error: tfluids.velocityUpdate BPROP dim ' .. dim)

    -- Now test and profile the CUDA version.
    profileAndTestCuda(tfluids.velocityUpdateForward,
                       'velocityUpdateForward_' .. dim .. 'd',
                       {UOurs, flags, pressure})
    local gradOutput = U:clone():uniform(0, 1)
    local gradP = pressure:clone()
    profileAndTestCuda(tfluids.velocityUpdateBackward,
                       'velocityUpdateBackward_' .. dim .. 'd',
                       {U, flags, pressure, gradOutput, gradP}, {5})
  end
end

function test.vorticityConfinement()
  for dim = 2, 3 do
    -- Load the input Manta data.
    local fn = dim .. 'd_buoyancy.bin'
    local _, U, flags, _, is3D = loadMantaBatch(fn)
    assertNotAllEqual(U)
    assertNotAllEqual(flags)

    assert(is3D == (dim == 3))
    -- Now load the output Manta data.
    fn = dim .. 'd_vorticityConfinement.bin'
    local _, UManta, flagsManta, _, is3D = loadMantaBatch(fn)
    assert(is3D == (dim == 3))
    assert(torch.all(torch.eq(flags, flagsManta)), 'flags changed!')

    -- Make sure this isn't a trivial velocity update (i.e. that velocities
    -- actually changed).
    assert((U - UManta):abs():max() > 1e-5, 'No velocities changed in Manta!')

    -- Perform our own velocity update calculation.
    -- A note here: it would seem like the vort confinement calculation is
    -- sensitive to float precision (likely due to the norm calls). We can do
    -- the test here in doubles (as the Manta sim did) and then convert to float
    -- for comparison.
    local UOurs = U:clone():double()  -- This is the divergent U.
    local strength = tfluids.getDx(flags)
    tfluids.vorticityConfinement(UOurs, flags:clone():double(), strength)
    local err = UManta - UOurs:float()
    mytester:assertlt(err:abs():max(), precision,
                      ('Error: tfluids.vorticityConfinement dim ' .. dim))
    
    -- Now test and profile the CUDA version.
    profileAndTestCuda(tfluids.vorticityConfinement, 
                       'vorticityConfinement_' .. dim .. 'd',           
                       {UOurs, flags, strength})
  end
end

function test.addBuoyancy()
  for dim = 2, 3 do
    -- Load the input Manta data.
    local fn = dim .. 'd_setWallBcs1.bin'
    local _, U, flags, density, is3D = loadMantaBatch(fn)
    assertNotAllEqual(U)  
    assertNotAllEqual(flags)
    assertNotAllEqual(density)

    assert(is3D == (dim == 3))
    -- Now load the output Manta data.
    fn = dim .. 'd_buoyancy.bin'
    local _, UManta, flagsManta, _, is3D = loadMantaBatch(fn)
    assert(is3D == (dim == 3))
    assert(torch.all(torch.eq(flags, flagsManta)), 'flags changed!')

    -- Make sure this isn't a trivial velocity update (i.e. that velocities
    -- actually changed). 
    assert((U - UManta):abs():max() > 1e-5, 'No velocities changed in Manta!')

    -- Perform our own velocity update calculation.
    local UOurs = U:clone()  -- This is the divergent U.
    local gStrength = tfluids.getDx(flags) / 4
    local gravity = torch.FloatTensor({1, 2, 3})
    if dim == 2 then
      gravity[3] = 0
    end
    gravity:div(gravity:norm()):mul(gStrength)
    local dt = 0.1
    tfluids.addBuoyancy(UOurs, flags:clone(), density:clone(), gravity, dt)
    local err = UManta - UOurs
    mytester:assertlt(err:abs():max(), precision,
                      ('Error: tfluids.addBuoyancy dim ' .. dim))

    -- Now test and profile the CUDA version.
    profileAndTestCuda(tfluids.addBuoyancy,
                       'addBuoyancy_' .. dim .. 'd',  
                       {UOurs, flags, density, gravity, dt})
  end
end

function test.emptyDomain()
  for dim = 2, 3 do
    local nbatch = torch.random(1, 4)
    local bnd = torch.random(1, 3)
    local width = torch.random(bnd * 2 + 1, 12)
    local height = torch.random(bnd * 2 + 1, 12)
    local depth = torch.random(bnd * 2 + 1, 12)
    if dim == 2 then
      depth = 1
    end

    local flags = torch.Tensor(nbatch, 1, depth, height, width)
    tfluids.emptyDomain(flags, dim == 3, bnd)
    
    local flagsGT = torch.Tensor(nbatch, 1, depth, height, width)
    flagsGT:fill(tfluids.CellType.TypeFluid)
    local obs = tfluids.CellType.TypeObstacle
    flagsGT[{{}, {}, {}, {}, {1, bnd}}]:fill(obs)
    flagsGT[{{}, {}, {}, {}, {width - bnd + 1, width}}]:fill(obs)
    flagsGT[{{}, {}, {}, {1, bnd}, {}}]:fill(obs)
    flagsGT[{{}, {}, {}, {height - bnd + 1, height}, {}}]:fill(obs)
    if dim == 3 then
      flagsGT[{{}, {}, {1, bnd}, {}, {}}]:fill(obs)
      flagsGT[{{}, {}, {depth - bnd + 1, depth}, {}, {}}]:fill(obs)
    end

    mytester:assert(torch.all(torch.eq(flagsGT, flags)),
                    'emptyDomain CPU error')
    
    -- Now test and profile on the GPU.
    profileAndTestCuda(tfluids.emptyDomain, 'emptyDomain_' .. dim .. 'd',
                       {flags, dim == 3, bnd}, {1})
  end
end

function test.flagsToOccupancy()
  for dim = 2, 3 do
    local nbatch = torch.random(1, 4)
    local width = torch.random(6, 12)
    local height = torch.random(6, 12)
    local depth = torch.random(6, 12)
    if dim == 2 then
      depth = 1
    end

    local occupancy = torch.Tensor(nbatch, 1, depth, height, width):random(0, 1)
    local flags = occupancy:clone():fill(0)

    local pocc = occupancy:data()
    local pflags = flags:data()
    local numel = occupancy:numel()

    local fluid = tfluids.CellType.TypeFluid
    local occ = tfluids.CellType.TypeObstacle
    for i = 0, numel - 1 do
      if pocc[i] == 0 then
        pflags[i] = fluid
      else
        pflags[i] = occ
      end
    end

    local occupancyOut = occupancy:clone():fill(-1)
    tfluids.flagsToOccupancy(flags, occupancyOut)
    mytester:assert(torch.all(torch.eq(occupancy, occupancyOut)),
                    'flagsToOccupancy CPU error')

    -- Try it in our module.
    local mod = tfluids.FlagsToOccupancy()
    local occupancyMod = mod:forward(flags)
    mytester:assert(torch.all(torch.eq(occupancy, occupancyMod)),
                    'flagsToOccupancy module error')

    -- Now test and profile on the GPU.
    profileAndTestCuda(tfluids.flagsToOccupancy,
                       'flagsToOccupancy_' .. dim .. 'd',
                       {flags, occupancyOut}, {2})
  end
end

function test.VolumetricUpSamplingNearest()
  local batchSize = torch.random(1, 5)
  local nPlane = torch.random(1, 5)
  local widthIn = torch.random(5, 8)
  local heightIn = torch.random(5, 8)
  local depthIn = torch.random(5, 8)
  local ratio = torch.random(2, 3)  -- We should probably test '1' as well...

  local module = tfluids.VolumetricUpSamplingNearest(ratio)

  local input = torch.rand(batchSize, nPlane, depthIn, heightIn, widthIn)
  local output = module:forward(input):clone()

  assert(output:dim() == 5)
  assert(output:size(1) == batchSize)
  assert(output:size(2) == nPlane)
  assert(output:size(3) == depthIn * ratio)
  assert(output:size(4) == heightIn * ratio)
  assert(output:size(5) == widthIn * ratio)

  local outputGT = torch.Tensor():resizeAs(output)
  for b = 1, batchSize do
    for f = 1, nPlane do
      for z = 1, depthIn * ratio do
        local zIn = math.floor((z - 1) / ratio) + 1
        for y = 1, heightIn * ratio do
          local yIn = math.floor((y - 1) / ratio) + 1
          for x = 1, widthIn * ratio do
            local xIn = math.floor((x - 1) / ratio) + 1
            outputGT[{b, f, z, y, x}] = input[{b, f, zIn, yIn, xIn}]
          end
        end
      end
    end
  end

  -- Note FPROP should be exact (it's just a copy).
  mytester:asserteq((output - outputGT):abs():max(), 0, 'error on fprop')

  -- Generate a valid gradInput (we'll use it to test the GPU implementation).
  local gradOutput = torch.rand(batchSize, nPlane, depthIn * ratio,
                                heightIn * ratio, widthIn * ratio);
  local gradInput = module:backward(input, gradOutput):clone()

  -- Perform the function on the GPU.
  if tfluids.withCUDA then
    module:cuda()
    local inputGPU = input:cuda()
    local outputGPU = module:forward(inputGPU):double()
    mytester:assertle((output - outputGPU):abs():max(), precision,
                      'error on GPU fprop')

    local gradInputGPU =
        module:backward(inputGPU, gradOutput:cuda()):double()
    mytester:assertlt((gradInput - gradInputGPU):abs():max(), precision * 10,
                      'error on GPU bprop')
  end

  -- Check BPROP is correct.
  module:double()
  local err = jac.testJacobian(module, input)
  mytester:assertlt(err, precision,
                    'error on bprop\nsize in:\n' .. tostring(input:size()) ..
                    '\nsize out:\b' .. tostring(module.output:size()))

  -- Profile the FPROP.
  local res = math.floor(128 / ratio)
  local input = torch.FloatTensor(4, 8, res, res, res):uniform(0, 1)
  local output = torch.FloatTensor(4, 8, res * ratio, res * ratio, res * ratio)
  profileAndTestCuda(tfluids.volumetricUpSamplingNearestForward,
                     'volumetricUpSamplingNearestForward',
                     {ratio, input, output}, {3}, false)
  
  -- Profile the BPROP.
  local gradOutput = output:clone():uniform(0, 1)
  local gradInput = input:clone()
  profileAndTestCuda(tfluids.volumetricUpSamplingNearestBackward,
                     'volumetricUpSamplingNearestBackward',
                     {ratio, input, gradOutput, gradInput}, {4}, false)
end

function test.solveLinearSystemPCG()
  for dim = 2, 3 do
    for _, precondType in pairs({'none', 'ilu0', 'ic0'}) do
      -- Load the test data before the solvePressure call.
      -- Note: you need to run manta/scenes/_testData.py to generate the
      -- data for this test.
      local fn = dim .. 'd_setWallBcs2.bin'
      local _, UDiv, flagsDiv, _, is3DDiv = loadMantaBatch(fn)
      assertNotAllEqual(UDiv)
      assertNotAllEqual(flagsDiv)
  
      -- Load the ground truth pressure after solvePressure in manta.
      fn = dim .. 'd_solvePressure.bin'
      local pManta, UManta, flags, rhsManta, is3D = loadMantaBatch(fn)
 
      assert((flagsDiv - flags):abs():max() == 0, 'Flags changed!')
      assert(is3D == (dim == 3), '3D boolean is inconsistent')
      assert(is3D == is3DDiv, '3D boolean is inconsistent (before/after solve)')
  
      -- Calculate the divergence (the RHS of the linear system).
      -- Note that 'our velocityDivergenceForward' == 'manta makeRhs'.
      local div = flags:clone()
      tfluids.velocityDivergenceForward(UDiv, flags, div)
 
      mytester:assertlt((rhsManta - div):abs():max(), precision,
                        'PCG: Our divergence (rhs) is wrong.')
  
      -- Note: no need to call setWallBcs as the test data already has had this
      -- called.
  
      -- Call the forward function. Note: solveLinearSystemPCG is only
      -- implemented in CUDA.
      local p = flags:clone():uniform(0, 1):cuda()
      local tol = 1e-5
      local maxIter = 1000
      local verbose = false
      local residual = tfluids.solveLinearSystemPCG(
          p, flags:cuda(), div:cuda(), dim == 3, tol, maxIter, precondType,
          verbose)
  
      -- This next test MIGHT fail, if we hit max_iter.
      mytester:assertlt(residual, tol * 2,
                        'PCG residual ' .. residual .. ' high')

      local isNan = p:ne(p)
      mytester:assertlt(isNan:sum(), 1, 'pressure contains nan values!')

      -- Remove any arbitrary constant from both ours and Manta's pressure.
      p = p:float()
      -- Pressure test removed for now. There are some examples with a very
      -- ill-posed A where Manta's pressure is arbitrarily different. Overall
      -- as long as div(U) is small, we don't care if our pressure is a little
      -- off.
      --[[
      local bsz = p:size(1)
      local pMean = p:view(bsz, -1):mean(2):view(bsz, 1, 1, 1, 1)
      p = p - pMean:expandAs(p)
      local pMantaMean = pManta:view(bsz, -1):mean(2):view(bsz, 1, 1, 1, 1)
      pManta = pManta - pMantaMean:expandAs(pManta)
      local pTol = 1e-2
      mytester:assertlt((p - pManta):abs():max(), pTol,
                        'PCG pressure error.')
      --]]
  
      -- Now calculate the velocity update using this new pressure and
      -- the subsequent divergence.
      local UNew = UDiv:clone()
      tfluids.velocityUpdateForward(UNew, flags, p)
      local UDivNew = flags:clone():uniform(0, 1)
      tfluids.velocityDivergenceForward(UNew, flags, UDivNew)
      mytester:assertlt(UDivNew:abs():max(), 1e-4,
                        'PCG divergence error after velocityupdate! dim ' ..
                        dim .. ' precondType ' .. precondType)
      mytester:assertlt((UNew - UManta):abs():max(), 1e-4,
                        'PCG velocity error after velocityUpdate! dim ' ..
                        dim .. ' precondType ' .. precondType)
    end  
  end
end

function test.solveLinearSystemJacobi()
  for dim = 2, 3 do
    local fn = dim .. 'd_setWallBcs2.bin'
    local _, UDiv, flagsDiv, _, is3DDiv = loadMantaBatch(fn)
    assertNotAllEqual(UDiv)
    assertNotAllEqual(flagsDiv)

    fn = dim .. 'd_solvePressure.bin'
    local pManta, UManta, flags, rhsManta, is3D = loadMantaBatch(fn)

    assert((flagsDiv - flags):abs():max() == 0, 'Flags changed!')
    assert(is3D == (dim == 3), '3D boolean is inconsistent')
    assert(is3D == is3DDiv, '3D boolean is inconsistent (before/after solve)')

    -- Calculate the divergence (the RHS of the linear system).
    -- Note that 'our velocityDivergenceForward' == 'manta makeRhs'.
    local div = flags:clone()
    tfluids.velocityDivergenceForward(UDiv, flags, div)
    mytester:assertlt((rhsManta - div):abs():max(), precision,
                      'Jacobi: our divergence (rhs) is wrong.')
    -- Note: no need to call setWallBcs as the test data already has had this
    -- called.

    -- Call the forward function. Note: solveLinearSystemJacobi is only
    -- implemented in CUDA.
    local p = flags:clone():uniform(0, 1):cuda()
    local pTol = 0
    local maxIter = 100000  -- It has VERY slow convergence. Run for long time.
    local verbose = false
    local residual = tfluids.solveLinearSystemJacobi(
        p, flags:cuda(), div:cuda(), dim == 3, pTol, maxIter, verbose)

    mytester:assertlt(residual, 1e-4, 'Jacobi residual ' .. residual .. ' high')
    p = p:float()
    local pPrecision = 1e-4
    if dim == 3 then
      pPrecision = 1e-2
    end
    -- Note: Jacobi takes a REALLY long to settle any non-zero pressure
    -- constant. This is because the relaxation has to settle everywhere.
    -- However, constant pressure is completely ignored during the velocity
    -- update (since we take grad(p)). Therefore we're free to subtract off
    -- any pressure mean from ours and manta's solution and still have a valid
    -- test.
    -- Pressure test removed for now. There are some examples with a very
    -- ill-posed A where Manta's pressure is arbitrarily different. Overall
    -- as long as div(U) is small, we don't care if our pressure is a little
    -- off.
    --[[
    local bsz = p:size(1)
    local pMean = p:view(bsz, -1):mean(2):view(bsz, 1, 1, 1, 1)
    p = p - pMean:expandAs(p)
    local pMantaMean = pManta:view(bsz, -1):mean(2):view(bsz, 1, 1, 1, 1)
    pManta = pManta - pMantaMean:expandAs(pManta)
    mytester:assertlt((p - pManta):abs():max(), pPrecision,
                      'Jacobi pressure error.')
    --]]

    -- Now calculate the velocity update using this new pressure and
    -- the subsequent divergence.
    local UNew = UDiv:clone()
    tfluids.velocityUpdateForward(UNew, flags, p)
    local UDivNew = flags:clone():uniform(0, 1)
    tfluids.velocityDivergenceForward(UNew, flags, UDivNew)
    mytester:assertlt(UDivNew:abs():max(), 1e-5,
                      'Jacobi divergence error after velocityupdate.')
    mytester:assertlt((UNew - UManta):abs():max(), 1e-4,
                      'Jacobi velocity error after velocityUpdate.')
  end
end 

function test.rectangularBlur()
  for dim = 2, 3 do
    local nbatch = torch.random(1, 3)
    local nchan = 3
    local zsize = 1
    if dim == 3 then
      zsize = 64
    end
    ysize = 65
    xsize = 66

    local blurRad = torch.random(1, 4)

    local src = torch.Tensor(nbatch, nchan, zsize, ysize, xsize):uniform(0, 1)
    local dst = src:clone():uniform(0, 1)  -- Fill it with random stuff.
    local is3D = dim == 3

    tfluids.rectangularBlur(src, blurRad, is3D, dst)

    -- Now use a Spatial/VolumetricConvolution stage to replicate the same.
    -- Our blur kernel clamps the edge values, we should do the same.
    local k = blurRad * 2 + 1
    local mod
    if dim == 2 then
      mod = nn.SpatialConvolution(1, 1, k, k, 1, 1)  -- fin, fout, kW/H, dW/H
    else
      mod = nn.VolumetricConvolution(1, 1, k, k, k, 1, 1, 1)
    end
    mod.weight:fill(1):div(mod.weight:sum())
    mod.bias:fill(0)
    
    local pad = k - 1
    
    local srcPad
    if dim == 2 then
      srcPad = torch.Tensor(nbatch, nchan, zsize, ysize + pad, xsize + pad)
    else
      srcPad = torch.Tensor(nbatch, nchan, zsize + pad, ysize + pad,
                            xsize + pad)
    end
    for z = 1, srcPad:size(3) do
      local zsrc = math.min(math.max(z - pad / 2, 1), src:size(3))
      for y = 1, srcPad:size(4) do
        local ysrc = math.min(math.max(y - pad / 2, 1), src:size(4))
        for x = 1, srcPad:size(5) do
          local xsrc = math.min(math.max(x - pad / 2, 1), src:size(5))
          srcPad[{{}, {}, z, y, x}]:copy(src[{{}, {}, zsrc, ysrc, xsrc}])
        end
      end
    end
    local dstGT = dst:clone():uniform(0, 1)
    for i = 1, nchan do
      if dim == 2 then
        dstGT[{{}, i}]:copy(mod:forward(srcPad[{{}, {i}, 1, {}, {}}]))
      else
        dstGT[{{}, i}]:copy(mod:forward(srcPad[{{}, {i}, {}, {}, {}}]))
      end
    end

    mytester:assertlt((dst - dstGT):abs():max(), precision, 'blur error')
  end
end

function test.signedDistanceField()
  for dim = 2, 3 do
    local batchSize = torch.random(1, 5)
    local widthIn = torch.random(16, 32)
    local heightIn = torch.random(16, 32)
    local depthIn = torch.random(16, 32)
    local searchRad = torch.random(1, 5)

    if dim == 2 then
      depthIn = 1
    end

    local flags = torch.rand(batchSize, 1, depthIn, heightIn, widthIn)
    -- Turn it into a {0, TypeObstacle} grid.
    flags = flags:gt(0.8):double():mul(tfluids.CellType.TypeObstacle)
    local dist = flags:clone():uniform(0, 1)
    local is3D = dim == 3

    tfluids.signedDistanceField(flags, searchRad, is3D, dist)
  
    -- Make sure the distances are correct.
    -- This is a pretty dumb test. It's just a reimplementation of the same code
    -- in lua.
    local obs = tfluids.CellType.TypeObstacle
    local distGT = dist:clone():uniform(0, 1)
    for b = 1, flags:size(1) do
      for z = 1, flags:size(3) do
        for y = 1, flags:size(4) do
          for x = 1, flags:size(5) do
            if flags[{b, 1, z, y, x}] == obs then
              distGT[{b, 1, z, y, x}] = 0
            else
              local distSq = searchRad * searchRad
              local zStart = math.max(1, z - searchRad)
              local zEnd = math.min(depthIn, z + searchRad)
              local yStart = math.max(1, y - searchRad)
              local yEnd = math.min(heightIn, y + searchRad)
              local xStart = math.max(1, x - searchRad)
              local xEnd = math.min(widthIn, x + searchRad)
              for zoff = zStart, zEnd do
                for yoff = yStart, yEnd do
                  for xoff = xStart, xEnd do
                    if flags[{b, 1, zoff, yoff, xoff}] == obs then
                      curDistSq = ((z - zoff) * (z - zoff) +  
                                   (y - yoff) * (y - yoff) +
                                   (x - xoff) * (x - xoff))
                      if curDistSq < distSq then
                        distSq = curDistSq
                      end
                    end
                  end
                end
              end
              distGT[{b, 1, z, y, x}] = math.sqrt(distSq)
            end
          end
        end
      end
    end
    local err = dist - distGT
    mytester:assertlt(err:abs():max(), precision, 'signedDistanceField error')

    -- Now test another flags grid, with a known (and easy) solution.
    -- Generate a bunch of single points in the flags grid.
    flags:fill(tfluids.CellType.TypeFluid)
    local npnts = 4
    local pntPos = {}
    for b = 1, batchSize do
      pntPos[b] = {}
      for i = 1, npnts do
        local pos =  torch.Tensor({torch.random(1, widthIn),
                                   torch.random(1, heightIn),
                                   torch.random(1, depthIn)})
        pntPos[b][i] = pos
        flags[{b, 1, pos[3], pos[2], pos[1]}] = tfluids.CellType.TypeObstacle
      end
    end
    -- Calculate what the GT output should be.
    for b = 1, flags:size(1) do
      for z = 1, flags:size(3) do
        for y = 1, flags:size(4) do
          for x = 1, flags:size(5) do
            local dist = searchRad
            for i = 1, npnts do
              local delta = torch.Tensor({x, y, z}) - pntPos[b][i]
              local cur_dist = delta:norm()
              if cur_dist < dist then
                dist = cur_dist
              end
            end
            distGT[{b, 1, z, y, x}] = dist
          end
        end
      end
    end
    -- Now test this against the tfluids function.
    tfluids.signedDistanceField(flags, searchRad, is3D, dist)
    err = dist - distGT
    mytester:assertlt(err:abs():max(), precision, 'signedDistanceField error 2')

    -- Now profile and test GPU version.
    profileAndTestCuda(tfluids.signedDistanceField,
                       'signedDistanceField dim ' .. dim,
                       {flags, searchRad, is3D, dist}, {4})
  end
end

-- Now run the test above
mytester:add(test)

function tfluids.test(tests, seed)
  dofile('../lib/ls.lua')
  dofile('../lib/load_manta_file.lua')  -- For torch.loadMantaFile()
  dofile('../lib/modules/inject_tensor.lua')

  local curDevice = cutorch.getDevice()
  gpuDevice = gpuDevice or 1
  cutorch.setDevice(gpuDevice)
  print('Testing on gpu device ' .. gpuDevice)
  print(cutorch.getDeviceProperties(gpuDevice))

  -- randomize stuff.
  local seed = seed or (1e5 * torch.tic())
  print('Seed: ', seed)
  math.randomseed(seed)
  torch.manualSeed(seed)
  mytester:run(tests)

  numTimes = 0
  for _, _ in pairs(times) do
    numTimes = numTimes + 1
  end

  if numTimes > 1 then
    print ''
    print('-----------------------------------------------------------------' ..
          '-------------')
    print('| Module - At resolution ' ..
          string.format('%-4d', profileResolution) ..
          '                                  | Speedup     |')
    print('-----------------------------------------------------------------' ..
          '-------------')
    for module, tm in pairs(times) do
      local str = string.format('| %-60s | %6.2f      |', module,
                                (tm.cpu / tm.gpu))
      print(str)
    end
    print('-----------------------------------------------------------------' ..
          '-------------')
  end

  return mytester
end

