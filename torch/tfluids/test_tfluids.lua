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

-- You can easily test specific units like this:
-- qlua -ltfluids -e "tfluids.test{'calcVelocityDivergenceCUDA'}"

-- Or to test everything:
-- qlua -ltfluids -e "tfluids.test()"

local cutorch = require('cutorch')

torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local test = torch.TestSuite()
local times = {}
local profileTimeSec = 1

function test.MoveImpulseWithinImage2D()
  local methods = {'euler', 'rk2'}
  for _, method in pairs(methods) do
    local d = 1
    local w = torch.random(5, 10)
    local h = torch.random(5, 10)
    local s = torch.zeros(d, h, w)
    -- Create an impulse.
    local u = torch.random(2, w - 1)
    local v = torch.random(2, h - 1)
    s[{1, v, u}] = 1
    -- Move the impulse to another location in the image by advecting it.
    local dx = torch.random(-u + 1, w - u)
    local dy = torch.random(-v + 1, h - v)
    -- Add a tiny offset so it is not EXACTLY in the center.
    local vel = torch.zeros(2, d, h, w)
    vel[1]:fill(dx + torch.rand(1)[1] * 2e-5 - 1e-5)
    vel[2]:fill(dy + torch.rand(1)[1] * 2e-5 - 1e-5)
    local dt = 1
    local geom = torch.zeros(d, h, w)
    local sM = s:clone():fill(0)
    tfluids.advectScalar(dt, s, vel, geom, sM, method)
    local uM = u + dx
    local vM = v + dy
    -- Make sure we did keep the ground truth location on the image.
    assert(vM >= 1 and vM <= h and uM >= 1 and uM <= w, 'Test logic error')
    mytester:eq(sM[{1, vM, uM}], 1, 1e-4, 'Bad (1) position!')
    sM[{1, vM, uM}] = 0
    -- Now the whole thing should be zeros.
    mytester:asserteq(sM:min(), 0, 'Bad position!')
    mytester:assertlt(sM:max(), 1e-4, 'Bad position!') 
  end
end

function test.ScalarAdvectionLinear()
  -- TODO(tompson): Finish this.
end

local function profileCuda(func, name, args)
  local tm = {}
  times[name] = tm

  -- Profile CPU.
  for key, value in pairs(args) do
    if torch.isTensor(value) then
      args[key] = value:float()
    end
  end
  local a = torch.Timer()
  local count = 0
  while a:time().real < profileTimeSec do
    count = count + 1
    func(unpack(args))
  end
  tm.cpu = a:time().real / count

  -- Profile GPU.
  for key, value in pairs(args) do
    if torch.isTensor(value) then
      args[key] = value:cuda()
    end
  end
  a:reset()
  while a:time().real < profileTimeSec do
    count = count + 1
    func(unpack(args))
  end   
  tm.gpu = a:time().real / count
end

function test.averageBorderCellsCUDA()
  -- TODO(tompson): Write a test of the forward function.

  -- Test that the float and cuda implementations are the same.
  local nchan = {2, 3}
  local d = {1, torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local case = {'2D', '3D'}

  for testId = 1, 2 do
    -- 2D and 3D cases.
    local geom = torch.rand(d[testId], h, w):gt(0.8):float()
    local input = torch.rand(nchan[testId], d[testId], h, w):float()
    local outputCPU = input:clone()

    -- Perform the function on the CPU.
    tfluids.averageBorderCells(input, geom, outputCPU)

    -- Perform the function on the GPU.
    local outputGPU = input:cuda()
    tfluids.averageBorderCells(input:cuda(), geom:cuda(), outputGPU)

    -- Compare the results.
    local maxErr = (outputCPU - outputGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'averageBorderCells CUDA ERROR ' .. case[testId])

    profileCuda(tfluids.averageBorderCells,
                'averageBorderCells' .. case[testId], {input, geom, outputCPU})
  end
end

function test.vorticityConfinementCUDA()
  -- TODO(tompson): Write a test of the forward function.

  -- Test that the float and cuda implementations are the same.
  local nchan = {2, 2, 3, 3}
  local d = {1, 1, torch.random(32, 64), torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local scale = torch.rand(1)[1]  -- in [0, 1]
  local dt = torch.rand(1)[1]  -- in [0, 1]
  local case = {'2D', '2DGEOM', '3D', '3DGEOM'}
  local incGeom = {false, true, false,  true}

  for testId = 1, #nchan do
    -- 2D and 3D cases.
    local geom
    if not incGeom[testId] then
      geom = torch.zeros(d[testId], h, w):float()
    else
      geom = torch.rand(d[testId], h, w):gt(0.8):float()
    end

    local U = torch.rand(nchan[testId], d[testId], h, w):float()
    local magCurl = geom:clone():fill(0)
    local curl
    if nchan[testId] == 2 then
      curl = geom:clone()
    else
      curl = U:clone()
    end
    local UCPU = U:clone()

    -- Perform the function on the CPU.
    tfluids.vorticityConfinement(dt, scale, UCPU, geom, curl, magCurl)

    -- Perform the function on the GPU.
    local UGPU = U:cuda()
    local curlGPU = curl:cuda():fill(0)
    local magCurlGPU = magCurl:cuda():fill(0)
    tfluids.vorticityConfinement(dt, scale, UGPU, geom:cuda(), curlGPU,
                                 magCurlGPU)

    -- Compare the results.
    local maxErr = (curlGPU:float() - curl):abs():max()
    mytester:assertlt(maxErr, precision,
                      ('vorticityConfinementZeroGeom CUDA curl ERROR ' ..
                       case[testId]))
    maxErr = (magCurlGPU:float() - magCurl):abs():max()
    mytester:assertlt(maxErr, precision,
                      ('vorticityConfinementZeroGeom CUDA magCurl ERROR ' ..
                       case[testId]))
    maxErr = (UCPU - UGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      ('vorticityConfinementZeroGeom CUDA ERROR ' ..
                       case[testId]))

    profileCuda(tfluids.vorticityConfinement,
                'vorticityConfinementZeroGeom' .. case[testId],
                {dt, scale, U, geom, curl, magCurl})
  end
end

function test.calcVelocityUpdateCUDA()
  -- NOTE: The forward function test is split between:
  -- torch/utils/test_calc_velocity_update.lua and
  -- torch/lib/modules/test_velocity_update.lua

  -- Test that the float and cuda implementations are the same.
  local batchSize = torch.random(1, 3)
  local nchan = {2, 3, 2, 3}
  local d = {1, torch.random(32, 64), 1, torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local case = {'2D', '3D', '2DMatchManta', '3DMatchManta'}
  local matchManta = {false, false, true, true}

  for testId = 1, 4 do
    -- 2D and 3D cases.
    local geom = torch.rand(batchSize, d[testId], h, w):gt(0.8):float()
    local p = torch.rand(batchSize, d[testId], h, w):float()
    local outputCPU =
        torch.rand(batchSize, nchan[testId], d[testId], h, w):float()

    -- Perform the function on the CPU.
    tfluids.calcVelocityUpdate(outputCPU, p, geom, matchManta[testId])

    -- Perform the function on the GPU.
    local outputGPU = outputCPU:clone():fill(math.huge):cuda()
    tfluids.calcVelocityUpdate(outputGPU, p:cuda(), geom:cuda(),
                               matchManta[testId])

    -- Compare the results.
    local maxErr = (outputCPU - outputGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'calcVelocityUpdate CUDA ERROR ' .. case[testId])

    -- Now test the backwards call.
    local gradOutput =
      torch.rand(batchSize, nchan[testId], d[testId], h, w):float()

    local gradPCPU = p:clone():fill(math.huge)
    tfluids.calcVelocityUpdateBackward(gradPCPU, p, geom, gradOutput,
                                       matchManta[testId])

    local gradPGPU = p:clone():fill(math.huge):cuda()
    tfluids.calcVelocityUpdateBackward(gradPGPU, p:cuda(), geom:cuda(),
                                       gradOutput:cuda(), matchManta[testId])

    maxErr = (gradPCPU - gradPGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'calcVelocityUpdateBackward CUDA ERROR ' .. case[testId])

    profileCuda(tfluids.calcVelocityUpdate,
                'calcVelocityUpdate' .. case[testId],
                {outputCPU, p, geom, matchManta[testId]})

    profileCuda(tfluids.calcVelocityUpdateBackward,
                'calcVelocityUpdateBackward' .. case[testId],
                {gradPCPU, p, geom, gradOutput, matchManta[testId]})
  end
end

--[[
function test.solveLinearSystemPCGCUDA()
  local batchSize = torch.random(1, 3)
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  for dim = 2, 3 do
    local d
    if dim == 3 then
      d = torch.random(32, 64)
    else
      d = 1
    end

    -- Create some random inputs.
    local geom = torch.rand(batchSize, d, h, w):gt(0.8):cuda()
    local p = torch.rand(batchSize, d, h, w):cuda()
    local U = torch.rand(batchSize, dim, d, h, w):cuda()

    -- Allocate the output tensor.
    local deltaU = torch.CudaTensor(batchSize, dim, d, h, w)

    -- Call the forward function.
    tfluids.solveLinearSystemPCG(deltaU, p, geom, U)

    -- TODO(kris): Check the output.
  end
end
--]]

function test.calcVelocityDivergenceCUDA()
  -- NOTE: The forward and backward function tests are in:
  -- torch/lib/modules/test_velocity_divergence.lua

  -- Test that the float and cuda implementations are the same.
  local batchSize = torch.random(1, 3)
  local nchan = {2, 3}
  local d = {1, torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local case = {'2D', '3D'}

  for testId = 1, 2 do
    local geom = torch.rand(batchSize, d[testId], h, w):gt(0.8):float()
    local U = torch.rand(batchSize, nchan[testId], d[testId], h, w):float()
    local UDivCPU = geom:clone():fill(0)
    
    -- Perform the function on the CPU.
    tfluids.calcVelocityDivergence(U, geom, UDivCPU)

    -- Perform the function on the GPU. 
    local UDivGPU = UDivCPU:clone():fill(math.huge):cuda()
    tfluids.calcVelocityDivergence(U:cuda(), geom:cuda(), UDivGPU)

    -- Compare the results.
    local maxErr = (UDivCPU - UDivGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'calcVelocityDivergence CUDA ERROR ' .. case[testId])
  
    -- Now test the backwards call.
    local gradOutput =
      torch.rand(batchSize, d[testId], h, w):float()
  
    local gradUCPU = U:clone():fill(math.huge)
    tfluids.calcVelocityDivergenceBackward(gradUCPU, U, geom, gradOutput)
  
    local gradUGPU = U:clone():fill(math.huge):cuda()
    tfluids.calcVelocityDivergenceBackward(gradUGPU, U:cuda(), geom:cuda(),
                                       gradOutput:cuda())

    maxErr = (gradUCPU - gradUGPU:float()):abs():max()
    mytester:assertlt(
        maxErr, precision, 'calcVelocityDivergenceBackward CUDA ERROR ' ..
        case[testId])

    profileCuda(tfluids.calcVelocityDivergence,
                'calcVelocityDivergence' .. case[testId], {U, geom, UDivCPU})
    
    profileCuda(tfluids.calcVelocityDivergenceBackward,
                'calcVelocityDivergenceBackward' .. case[testId],
                {gradUCPU, U, geom, gradOutput})
  end
end 

function test.interpField()
  local d = torch.random(8, 16)
  local h = torch.random(8, 16)
  local w = torch.random(8, 16)

  local geom = torch.zeros(d, h, w)
  local field = torch.rand(d, h, w)

  -- Center of the first cell.
  local pos = torch.Tensor({0, 0, 0})
  local val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, 1, 1}], 'Bad interp value')

  -- Center of the last cell.
  pos = torch.Tensor({w - 1, h - 1, d - 1})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{-1, -1, -1}], 'Bad interp value')
  
  -- The corner of the grid should also be the center of the first cell.
  pos = torch.Tensor({-0.5, -0.5, -0.5})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, 1, 1}], 'Bad interp value')

  -- The right edge of the first cell should be the average of the two.
  pos = torch.Tensor({0.5, 0, 0})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, 1, {1, 2}}]:mean(), 'Bad interp value')

  -- The top edge of the first cell should be the average of the two.
  pos = torch.Tensor({0, 0.5, 0})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{1, {1, 2}, 1}]:mean(), 'Bad interp value')

  -- The back edge of the first cell should be the average of the two.
  pos = torch.Tensor({0, 0, 0.5})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{{1, 2}, 1, 1}]:mean(), 'Bad interp value')

  -- The corner of the first cell should be the average of all the neighbours.
  pos = torch.Tensor({0.5, 0.5, 0.5})
  val = tfluids.interpField(field, geom, pos)
  mytester:asserteq(val, field[{{1, 2}, {1, 2}, {1, 2}}]:mean(),
                    'Bad interp value')

  -- TODO(tompson,kris): Is this enough test cases?
end

function test.setObstacleBcs()
  local nchan = {2, 3}
  local d = {1, torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)

  for testId = 1, 2 do
    local twoDim = nchan[testId] == 2
    local geom = torch.zeros(d[testId], h, w)
    local U = torch.rand(nchan[testId], d[testId], h, w)

    -- Make a contiguous chunk of geometry.
    local istart = 16
    local iend = 20
    local i0 = (istart - 1) - 0.5 - 1e-6  -- The geom face in 0-index coords.
    local i1 = (iend - 1) + 0.5 + 1e-6  -- The geom face in 0-index coords.
    local ic = (iend + istart) * 0.5 - 1  -- Center in 0-index coords.
    if not twoDim then
      geom[{{16, 20}, {16, 20}, {16, 20}}]:fill(1)
    else
      geom[{1, {16, 20}, {16, 20}}]:fill(1)
    end

    -- Set the internal U values.
    tfluids.setObstacleBcs(U, geom)

    -- Now interpolate a value at one of the geometry faces and make sure
    -- the velocity component is zero.
    -- Left face.
    local pos = torch.Tensor({i0, ic, ic})  -- Recall: 0-indexed.
    if twoDim then
      pos[3] = 0
    end
    local Ux = tfluids.interpField(U[1], geom, pos)
    mytester:assertlt(math.abs(Ux), 1e-5, 'Bad face velocity value ' .. testId)

    -- Right face.
    pos = torch.Tensor({i1, ic, ic})  -- Recall: 0-indexed.
    if twoDim then
      pos[3] = 0
    end
    Ux = tfluids.interpField(U[1], geom, pos)
    mytester:assertlt(math.abs(Ux), 1e-5, 'Bad face velocity value ' .. testId)

    -- Bottom face.
    pos = torch.Tensor({ic, i0, ic})  -- Recall: 0-indexed.
    if twoDim then
      pos[3] = 0
    end
    local Uy = tfluids.interpField(U[2], geom, pos)
    mytester:assertlt(math.abs(Uy), 1e-5, 'Bad face velocity value ' .. testId)

    -- Top face.
    pos = torch.Tensor({ic, i1, ic})  -- Recall: 0-indexed.
    if twoDim then
      pos[3] = 0
    end
    Uy = tfluids.interpField(U[2], geom, pos)
    mytester:assertlt(math.abs(Uy), 1e-5, 'Bad face velocity value ' .. testId)

    if not twoDim then
      -- Bottom face.
      pos = torch.Tensor({ic, ic, i0})  -- Recall: 0-indexed.
      local Uz = tfluids.interpField(U[3], geom, pos)
      mytester:assertlt(math.abs(Uz), 1e-5,
                        'Bad face velocity value ' .. testId)

      -- Bottom face.
      pos = torch.Tensor({ic, ic, i1})  -- Recall: 0-indexed.
      Uz = tfluids.interpField(U[3], geom, pos)
      mytester:assertlt(math.abs(Uz), 1e-5,
                        'Bad face velocity value ' .. testId)
    end

    -- TODO(tompson): Is this enough test cases?
  end
end

function test.setObstacleBcsCUDA()
  -- TODO(tompson): Write a test of the forward function.

  -- Test that the float and cuda implementations are the same.
  local nchan = {2, 3}
  local d = {1, torch.random(32, 64)}
  local w = torch.random(32, 64)
  local h = torch.random(32, 64)
  local case = {'2D', '3D'}

  for testId = 1, 2 do
    -- 2D and 3D cases.
    local geom = torch.rand(d[testId], h, w):gt(0.8):float()
    local U = torch.rand(nchan[testId], d[testId], h, w):float()
    local UCPU = U:clone()

    -- Perform the function on the CPU.
    tfluids.setObstacleBcs(UCPU, geom)

    -- Perform the function on the GPU.
    local UGPU = U:cuda()
    tfluids.setObstacleBcs(UGPU, geom:cuda())

    -- Compare the results.
    local maxErr = (UCPU - UGPU:float()):abs():max()
    mytester:assertlt(maxErr, precision,
                      'setObstacleBcs CUDA ERROR ' .. case[testId])
    
    profileCuda(tfluids.setObstacleBcs, 'setObstacleBcs2D' .. case[testId],
                {U, geom})
  end
end

-- Now run the test above
mytester:add(test)

function tfluids.test(tests, seed, gpuDevice)
  local curDevice = cutorch.getDevice()
  -- By default don't test on the primary device.
  gpuDevice = gpuDevice or 1
  cutorch.setDevice(gpuDevice)
  print('Testing on gpu device ' .. gpuDevice)
  print(cutorch.getDeviceProperties(gpuDevice))

  -- randomize stuff.
  local seed = seed or (1e5 * torch.tic())
  print('Seed: ', seed)
  math.randomseed(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  mytester:run(tests)

  print ''
  print('-----------------------------------------------------------------' ..
        '-------------')
  print('| Module                                                       ' ..
        '| Speedup     |')
  print('-----------------------------------------------------------------' ..
        '-------------')
  for module, tm in pairs(times) do
    local str = string.format('| %-60s | %6.2f      |', module,
                              (tm.cpu / tm.gpu))
    print(str)
  end
  print('-----------------------------------------------------------------' ..
        '-------------')

  cutorch.setDevice(curDevice)

  return mytester
end

