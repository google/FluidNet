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

local LerpCriterion, parent = torch.class('nn.LerpCriterion', 'nn.Criterion')

function LerpCriterion:__init(crit1, crit2, lambda, sizeAverage)
   parent.__init(self)
   self.crit1 = crit1
   self.crit2 = crit2
   self.lambda = lambda
   self.sizeAverage = sizeAverage
   self.crit1.sizeAverage = self.sizeAverage
   self.crit2.sizeAverage = self.sizeAverage
end

function LerpCriterion:updateOutput(input, target)
  self.output = (1.0 - self.lambda) * self.crit2:forward(input, target) + self.lambda * self.crit1:forward(input, target)
  return self.output
end

function LerpCriterion:updateGradInput(input, target)
  local gradInput1 = self.crit1:backward(input, target)
  local gradInput2 = self.crit2:backward(input, target)
  self.gradInput:resizeAs(input):fill(self.lambda)
  self.gradInput:cmul(gradInput1)
  self.gradInput:add(1.0 - self.lambda, gradInput2)
  return self.gradInput
end
