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

-- This is a pretty messy top level include script. Note that not all
-- libraries are needed for every simulation run (however a warning will be
-- printed anyway if they cannot be loaded).

local torch = require('torch')

dofile('lib/load_package_safe.lua')

local cutorch = torch.loadPackageSafe('cutorch')
torch.loadPackageSafe('cunn')
require 'image'
local optim = require('optim')
require 'gnuplot'
local cudnn = torch.loadPackageSafe('cudnn')
if cudnn ~= nil then
  -- cudnn.benchmark = true
  cudnn.fastest = true  -- Use this instead of 'benchmark' if unstable.
end
require 'nngraph'
require 'tfluids'
torch.loadPackageSafe('qt')
torch.loadPackageSafe('qttorch')
torch.loadPackageSafe('qtwidget')
torch.loadPackageSafe('qtuiloader')
torch.loadPackageSafe('mattorch')
torch.loadPackageSafe('torchzlib')

dofile('lib/fix_file_references.lua')

-- makeGlobal will add the local variable to the global namespace without
-- causing the strict library to throw an error or print a warning. It should
-- be very sparsely used.
function torch.makeGlobal(varName, var)
  rawset(_G, varName, var)
end

dofile('lib/misc_tools.lua')
dofile('lib/ls.lua')
dofile('lib/pbar.lua')
dofile('lib/load_manta_file.lua')
dofile('lib/data_binary.lua')
dofile('lib/run_epoch.lua')
dofile('lib/model.lua')
dofile('lib/model_utils.lua')
dofile('lib/parse_args.lua')
dofile('lib/modules/spatial_subtractive_normalization_batch.lua')
dofile('lib/modules/spatial_divisive_normalization_batch.lua')
dofile('lib/modules/spatial_contrastive_normalization_batch.lua')
dofile('lib/modules/lerp_criterion.lua')
dofile('lib/modules/select_fluid_input.lua')
dofile('lib/modules/spatial_divergence.lua')
dofile('lib/modules/spatial_finite_elements.lua')
dofile('lib/modules/volumetric_divergence.lua')
dofile('lib/modules/volumetric_finite_elements.lua')
dofile('lib/modules/fluid_criterion.lua')
dofile('lib/modules/weighted_flat_mse_criterion.lua')
dofile('lib/modules/residual_layer.lua')
dofile('lib/modules/variance.lua')
dofile('lib/modules/standard_deviation.lua')
dofile('lib/default_conf.lua')
dofile('lib/load_set.lua')
dofile('lib/simulate.lua')
dofile('lib/modules/spatial_convolution_upsample.lua')
dofile('lib/modules/volumetric_convolution_upsample.lua')
dofile('lib/modules/mse_si_criterion.lua')
dofile('lib/modules/apply_scale.lua')
dofile('lib/calc_stats.lua')
if optim.adam == nil then
  dofile('lib/adam.lua')
end
if optim.rmsprop == nil then
  dofile('lib/rmsprop.lua')
end
dofile('lib/queue.lua')
dofile('lib/data_parallel.lua')
dofile('lib/logger.lua')
dofile('lib/calc_flops.lua')

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(8)
torch.manualSeed(1)
if cutorch ~= nil then
  cutorch.manualSeed(1)
end
math.randomseed(1)

if math.mod == nil then
  math.mod = math.fmod
end

torch.loadPackageSafe('strict')  -- Must come last
