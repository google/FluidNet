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

local function scriptPath()
  local str = debug.getinfo(2, "S").source:sub(2)
  str = str:match("(.*/)")
  if str == nil then
    return './'
  else
    return str
  end
end
local path = scriptPath() .. "/"

local cunn = require("cunn")
local tfluids = require("tfluids")

-- Load the modules.
dofile(path .. "apply_scale.lua")
dofile(path .. "fluid_criterion.lua")
dofile(path .. "inject_tensor.lua")
dofile(path .. "spatial_convolution_upsample.lua")
dofile(path .. "spatial_divergence.lua")
dofile(path .. "spatial_finite_elements.lua")
dofile(path .. "standard_deviation.lua")
dofile(path .. "variance.lua")
dofile(path .. "volumetric_convolution_upsample.lua")
dofile(path .. "volumetric_divergence.lua")
dofile(path .. "volumetric_finite_elements.lua")
dofile(path .. "weighted_flat_mse_criterion.lua")
dofile(path .. "mse_si_criterion.lua")

-- Run the tests.
dofile(path .. "test_apply_scale.lua")
dofile(path .. "test_fluid_criterion.lua")
dofile(path .. "test_mse_si_criterion.lua")
dofile(path .. "test_spatial_convolution_upsample.lua")
dofile(path .. "test_spatial_divergence.lua")
dofile(path .. "test_spatial_finite_elements.lua")
dofile(path .. "test_standard_deviation.lua")
dofile(path .. "test_variance.lua")
dofile(path .. "test_volumetric_convolution_upsample.lua")
dofile(path .. "test_volumetric_divergence.lua")
dofile(path .. "test_volumetric_finite_elements.lua")
dofile(path .. "test_weighted_flat_mse_criterion.lua")

print("ALL TESTS FINISHED!")
