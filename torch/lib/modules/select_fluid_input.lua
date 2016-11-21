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

-- You will need to dofile('velocity_divergence.lua') and
-- dofile('velocity_divergence.lua').

local SelectFluidInput, parent =
  torch.class('nn.SelectFluidInput', 'nn.Module')

function SelectFluidInput:__init(indices)
  parent.__init(self)
  self.indices = indices
end

--forward()
function SelectFluidInput:updateOutput(input)
  local batchSize, numImages, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  --print("batchSize: " .. batchSize .. " numImages: " .. numImages .. " H: " .. H .. " W: " .. W)
  self.output:resize(batchSize, #self.indices, H, W):zero()
  for i, k in ipairs(self.indices) do
    self.output[{ {}, i, {}, {} }] = input[{ {}, k, {}, {} }]
  end
  
  return self.output
end

--backward()
function SelectFluidInput:updateGradInput(input, gradOutput)
  -- parent constructor always allocates empty self.gradInput.
  -- TODO(kris): This needs to zero out the dims that are not selected.

   self.gradInput:resizeAs(input)
   self.gradInput:copy(input)
   return self.gradInput
end

