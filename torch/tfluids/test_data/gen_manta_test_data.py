# Copyright 2016 Google Inc, NYU.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We will simply create a pseudo random velocity field (using wavelet
# turbulence) and use this to advect a scalar and a velocity field.

from manta import *
import os, shutil, math, sys, random

# dimension two/three d
dim = 2
bWidth = 2
ores = 128 + 2 * bWidth
# We advect in manta at a very large velocity then downsample it. This helps
# account for the difference between 0.5 pixel shift due to MAC grid sampling. 
upscaleFactor = 1
res = ores * upscaleFactor

gs = vec3(res, res, res)
gsSmall = vec3(ores, ores, ores)
if (dim == 2):
  gs.z = 1  # 2D
  gsSmall.z = 1

random.seed(1945)
fixedSeed = 0

setDebugLevel(10)  # Print like crazy!

sm = Solver(name='main', gridSize = gs, dim=dim)
sm.timestep = 0.033
smSmall = Solver(name='small', gridSize=gsSmall, dim=dim)
smSmall.timestep = 0.033

directory = "output/"
tt = 0.0

# Create the big grid.
flags = sm.create(FlagGrid)
vel = sm.create(MACGrid)
velVec3 = sm.create(VecGrid)
density = sm.create(RealGrid)

flags.initDomain(boundaryWidth = bWidth * upscaleFactor)
flags.fillGrid() 
setOpenBound(flags, bWidth, 'xXyY', FlagOutflow | FlagFluid)

flagsSmall = smSmall.create(FlagGrid)
velSmallVec3 = smSmall.create(VecGrid)
densitySmall = smSmall.create(RealGrid)

flags.initDomain(boundaryWidth = bWidth)
flags.fillGrid()
setOpenBound(flags, bWidth, 'xXyY', FlagOutflow | FlagFluid)

vel.clear()
velVec3.clear()
density.clear()
velSmallVec3.clear()
densitySmall.clear()

# Create a pseudo-random velocity field. Note: manta does NOT handle
# applyNoiseVec3 to MAC grids properly. The last element in each MAC dimension
# is filled with garbage. We have to create the noise in a Grid<Vec3> first than
# resample to MACGrid.
noise = sm.create(NoiseField, loadFromFile=True)
posScale = 20 / math.pow(upscaleFactor, 1 / 3)
noise.posScale = vec3(posScale, posScale, posScale)
noise.clamp = True
noise.clampNeg = -100
noise.clampPos = 100
noise.valScale = 1
noise.valOffset = 0.075
noise.timeAnim = 0.3
scale = 2
applyNoiseVec3(flags=flags, target=velVec3, noise=noise, scale=scale)
getMACFromVec3(target=vel, source=velVec3)

# Also create a random scalar field.
dnoise = sm.create(NoiseField, loadFromFile=True)
posScale = 10 / math.pow(upscaleFactor, 1 / 3)
dnoise.posScale = vec3(posScale, posScale, posScale)
testall = sm.create(RealGrid)
testall.setConst(-1.)
scale = 4
addNoise(flags=flags, density=density, noise=dnoise, sdf=testall, scale=scale)

# Downsample the high res grid (note you have to scale the velocities for the
# advection result to match on the low res version).
interpolateGridVec3(target=velSmallVec3, source=velVec3, orderSpace=2)
interpolateGrid(target=densitySmall, source=density, orderSpace=2)

# Save the field values BEFORE advection at the lower resolution.
directory = "./2d/"
filename = "255_pre_advect.bin"
fullFilename = directory + "/" + filename
writeOutSimVec3(fullFilename, 0, velSmallVec3, densitySmall)

# Perform the advection (on high res grid).
advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, openBounds=True,
                   boundaryWidth=bWidth) 
advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2, openBounds=True,
                   boundaryWidth=bWidth)

# Convert from MAC to Vec3.
getVec3FromMAC(target=velVec3, source=vel)

# Downsample the high res grid.
interpolateGridVec3(target=velSmallVec3, source=velVec3, orderSpace=2)
interpolateGrid(target=densitySmall, source=density, orderSpace=2)

filename = "255_post_advect.bin"
fullFilename = directory + "/" + filename
writeOutSimVec3(fullFilename, 0, velSmallVec3, densitySmall)

print("All done. Test data is in %s." % (directory))

