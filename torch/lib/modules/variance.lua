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

-- Calculate the SAMPLE variance of the input.
-- var = sigma^2 = sum(X - mean)^2 / (n - 1)
--     = (n * sum(x_i^2) - (sum(x_i))^2) / (n * (n - 1))
--
-- FYI: I used the following maxima command to calculate the partial
-- derivative (because I'm lazy):
-- >> derivative(((x_1^2 + x_2^2 + x_3^2 + x_4^2) * N - 
-- >>   (x_1 + x_2 + x_3 + x_4)^2) / (N * (N - 1)), x_1);
-- 
--                      2 x_1 N - 2 (x_4 + x_3 + x_2 + x_1)
-- (%o1)                -----------------------------------
--                                   (N - 1) N
--
-- Note: unlike torch.var(x, dim) we do not remove the dimension that we
-- calculate the variance over. Instead we leave it around as a unary dimension
-- so that the number of input dims is the same as the number of output dims.
--
-- TODO(tompson): This module is pretty memory inefficient.

local Variance, parent = torch.class('nn.Variance', 'nn.Module')

function Variance:__init(dim)
  parent.__init(self)
  self._inSq = torch.Tensor()  -- Temp space
  self._inSum = torch.Tensor()  -- Temp space
  self._inSumSq = torch.Tensor()  -- Temp space
  self._dim = dim
end

function Variance:updateOutput(input)
  self._inSq:resizeAs(input)
  self._inSq:copy(input)
  self._inSq:cmul(self._inSq)

  assert(self._dim <= input:dim())
  local odim = {}
  for i = 1, input:dim() do
    if i == self._dim then
      odim[i] = 1
    else
      odim[i] = input:size(i)
    end
  end

  self.output:resize(unpack(odim))

  local n = input:size(self._dim)
  assert(n > 1, 'Sample variance requires more than one sample.')
  
  torch.sum(self.output, self._inSq, self._dim)
  self.output:mul(n)  -- n * sum(x_i^2)

  self._inSum:resize(unpack(odim))
  torch.sum(self._inSum, input, self._dim)
  self._inSumSq:resize(unpack(odim))
  self._inSumSq:copy(self._inSum):cmul(self._inSum)  -- (sum(x_i))^2

  self.output:add(-1, self._inSumSq) -- n * sum(x_i^2) - (sum(x_i))^2
  self.output:div(n * (n - 1))

  return self.output 
end

function Variance:updateGradInput(input, gradOutput)
  -- The gradient of var(x) w.r.t x_i is:
  -- ((2 * x_i * n) - 2 * sum(x)) / (n * (n - 1))

  local n = input:size(self._dim)
  assert(n > 1, 'Sample variance requires more than one sample.')

  self.gradInput:resizeAs(input)
  self.gradInput:copy(input):mul(2 * n)
  self.gradInput:add(-2, self._inSum:expandAs(input))
  self.gradInput:div(n * (n - 1))

  self.gradInput:cmul(gradOutput:expandAs(input))
  
  return self.gradInput
end

function Variance:accGradParameters(input, gradOutput, scale)
  -- No learnable parameters.
end

function Variance:accUpdateGradParameters(input, gradOutput, lr)
  -- No learnable parameters.
end

function Variance:type(type)
  parent.type(self, type)
  return self
end

