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

-- This is a implementation of David Eigen's scale invariant criterion.
--
-- D(x, y) = 1/n * sum_i ((x - y)^2) - 1/n^2 * (sum_i (x - y))^2
--         = 1/n * MSE - 1/n^2 * (Sum(error))^2
--
-- i.e. This is actually minimizing the empirical variance of the per-dimension
-- error (although David does not describe it this way). The network still needs
-- to predict the correct scale, only that all scales are weighted equally.
--
-- This module assumes sizeAverage == true.
--
-- The input is 5D by default (batch x f x d x h x w).

local MSESICriterion, parent = torch.class('nn.MSESICriterion', 'nn.Criterion')

function MSESICriterion:__init(numDimsNonBatch)
  parent.__init(self)

  self._numDimsNonBatch = numDimsNonBatch
  assert(self._numDimsNonBatch >= 1)
  self._mse = nn.MSECriterion()
  self._mse.sizeAverage = false  -- We will do the normalization.

  self._sumSq = nn.Sequential()
  self._sumSq:add(nn.CSubTable())  -- input - target
  for i = self._numDimsNonBatch + 1, 2, -1 do
    self._sumSq:add(nn.Sum(i))
  end
  self._sumSq:add(nn.Power(2))  -- squared
  self._sumSq:add(nn.Sum(1))  -- batch
end

function MSESICriterion:updateOutput(input, target)
  assert(input:dim() == self._numDimsNonBatch + 1 and
         target:dim() == self._numDimsNonBatch + 1)

  local n = input:numel()
  local mseLoss = self._mse:forward(input, target) / n
  local sumSqLoss = self._sumSq:forward({input, target}):squeeze() / (n * n)

  self.output = mseLoss + sumSqLoss
  return self.output
end

function MSESICriterion:updateGradInput(input, target)
  assert(input:dim() == self._numDimsNonBatch + 1 and
         target:dim() == self._numDimsNonBatch + 1)

  self.gradInput:resizeAs(input)

  local n = input:numel()
  local mseGradInput = self._mse:updateGradInput(input, target):mul(1 / n)

  local sumSqGradInput = self._sumSq:updateGradInput(
      {input, target}, torch.ones(1))[1]:mul(1 / (n * n))

  self.gradInput:copy(mseGradInput):add(sumSqGradInput)
  return self.gradInput
end

function MSESICriterion:type(type)
  parent.type(self, type)
  self._sumSq:type(type)
  self._mse:type(type)
  return self
end
