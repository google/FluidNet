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

-- Calculate the input STD using sample variance.

local StandardDeviation, parent = torch.class('nn.StandardDeviation',
                                              'nn.Module')

function StandardDeviation:__init(dim)
  parent.__init(self)
  self.modules = {}
  self.modules[1] = nn.Sequential()
  self.modules[1]:add(nn.Variance(dim))
  self.modules[1]:add(nn.Sqrt())
  self.output = self.modules[1].output
  self._dim = dim
end

function StandardDeviation:updateOutput(input)
  self.output = self.modules[1]:updateOutput(input)
  return self.output 
end

function StandardDeviation:updateGradInput(input, gradOutput)
  self.gradInput = self.modules[1]:updateGradInput(input, gradOutput)
  return self.gradInput
end

function StandardDeviation:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function StandardDeviation:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function StandardDeviation:type(type)
  parent.type(self, type)
  return self
end

