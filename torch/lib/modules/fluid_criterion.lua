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

-- This criterion implements the criterion of fluid_criterion.pdf for 2D and 3D
-- fluid fields.
--
-- The module assumes that the model output is a table of size 2:
-- output[1] = p (batch x 1 x depth x height x width)
-- output[2] = U (batch x 2/3 x depth x height x width)
--
-- It also assumes the target is a table of size 2:
-- target[1] = pTarget (batch x 1 x depth x height x width)
-- target[2] = UTarget (batch x 2/3 x depth x height x width)
-- target[3] = geom (batch x 1 x depth x height x width)
--
-- So, for 2D fields, the size(2) of U is 2 and the depth is assumed to be 1.
--
-- If scaleInvariant == true, then we'll use David Eigen's scale invariant
-- criterion.

local FluidCriterion, parent = torch.class('nn.FluidCriterion', 'nn.Criterion')

function FluidCriterion:__init(pLambda, uLambda, divLambda, scaleInvariant)
  parent.__init(self)

  if scaleInvariant == nil then
    scaleInvariant = false
  end

  self.scaleInvariant = scaleInvariant
  self.pLambda = pLambda
  self.uLambda = uLambda
  self.divLambda = divLambda

  if not self.scaleInvariant then
    -- Create the sub criterion for each term in the obj function.
    -- 1. Pressure loss.
    self._pLoss = nn.MSECriterion()
    -- 2. Velocity loss.
    self._uLoss = nn.MSECriterion()
    -- 3. Divergence loss.
    self._divLoss = nn.MSECriterion()
  else
    self._pLoss = nn.MSESICriterion(4)  -- numNonBatchDim
    self._uLoss = nn.MSESICriterion(4)
    self._divLoss = nn.MSESICriterion(3)
  end

  -- Network to calculate velocity divergence.
  self._divNetwork = nn.VelocityDivergence()

  self.sizeAverage = true

  self._divUTarget = torch.Tensor()
  self.gradInput = {torch.Tensor(), torch.Tensor()}  -- pGrad, UGrad.
end

function FluidCriterion:getTargets(target)
  assert(torch.type(target) == 'table' and #target == 3)
  
  local pTarget = target[1]
  local UTarget = target[2]
  local geom = target[3]

  assert(pTarget:dim() == 5 and UTarget:dim() == 5)
  local twoDim = UTarget:size(2) == 2
  assert(pTarget:size(1) == UTarget:size(1))  -- nBatch
  assert(pTarget:size(2) == 1)
  if not twoDim then
    assert(UTarget:size(2) == 3)
  end
  assert(pTarget:size(3) == UTarget:size(3))  -- zdim
  if twoDim then
    assert(pTarget:size(3) == 1)
  end
  assert(pTarget:size(4) == UTarget:size(4))  -- ydim
  assert(pTarget:size(5) == UTarget:size(5))  -- xdim
  assert(pTarget:isSameSizeAs(geom))

  -- Remove the "feature" dimension from geom.
  geom = geom:select(2, 1)

  -- Create a dummy tensor for divergence that is all zeros (we'll use MSE
  -- to a zero target as the criterion).
  if not self._divUTarget:isSameSizeAs(geom) then
    self._divUTarget:resizeAs(geom)
    self._divUTarget:fill(0)  -- if branch avoids spurious fill(0) calls.
  end

  return pTarget, UTarget, self._divUTarget, twoDim, geom
end

function FluidCriterion:getInputs(input)
  assert(torch.type(input) == 'table' and #input == 2)

  local pPred = input[1]
  local UPred = input[2]

  assert(pPred:dim() == 5 and UPred:dim() == 5)
  local twoDim = UPred:size(2) == 2
  assert(pPred:size(1) == UPred:size(1))  -- nBatch
  assert(pPred:size(2) == 1)
  if not twoDim then
    assert(UPred:size(2) == 3)
  end
  assert(pPred:size(3) == UPred:size(3))
  if twoDim then
    assert(pPred:size(3) == 1)
  end
  assert(pPred:size(4) == UPred:size(4))
  assert(pPred:size(5) == UPred:size(5))

  return pPred, UPred, twoDim
end

function FluidCriterion:updateOutput(input, target)
  local pPred, UPred, twoDim = self:getInputs(input)
  local pTarget, UTarget, divUTarget, twoDimTarget, geom =
      self:getTargets(target)
  assert(twoDim == twoDimTarget)

  -- Propagate sizeAverage value to sub-modules.
  self._pLoss.sizeAverage = self.sizeAverage
  self._uLoss.sizeAverage = self.sizeAverage
  self._divLoss.sizeAverage = self.sizeAverage

  -- FPROP each loss.
  if self.pLambda > 0 then
    self.pLoss = self.pLambda *
        self._pLoss:updateOutput(pPred, pTarget)
  else
    self.pLoss = 0
  end
  if self.uLambda > 0 then
    self.uLoss = self.uLambda *
        self._uLoss:updateOutput(UPred, UTarget)
  else
    self.uLoss = 0
  end
  if self.divLambda > 0 then
    -- Calculate the divergence of the predicted velocity.
    local divUPred = self._divNetwork:forward({UPred, geom})
    self.divLoss = self.divLambda *
        self._divLoss:updateOutput(divUPred, divUTarget)
  else
    self.divLoss = 0
  end

  -- Calculate the total loss.
  self.output = self.pLoss + self.uLoss + self.divLoss

  return self.output
end

function FluidCriterion:updateGradInput(input, target)
  -- Assume updateOutput has already been called (a standard torch assumption).
  local pPred, UPred, twoDim = self:getInputs(input)
  local pTarget, UTarget, divUTarget, twoDimTarget, geom =
      self:getTargets(target)
  local divUPred = self._divNetwork.output

  assert(twoDim == twoDimTarget)

  -- Calculate the gradInput for each loss.
  local pGradInput
  if self.pLambda > 0 then
    pGradInput = self._pLoss:updateGradInput(pPred, pTarget)
    pGradInput:mul(self.pLambda)
  end
  local uGradInput
  if self.uLambda > 0 then
    uGradInput = self._uLoss:updateGradInput(UPred, UTarget)
    uGradInput:mul(self.uLambda)
  end
  local divGradInput
  if self.divLambda > 0 then
    divGradInput = self._divLoss:updateGradInput(divUPred, divUTarget)
    divGradInput:mul(self.divLambda)
    -- BPROP through the divergence network.
    divGradInput = self._divNetwork:updateGradInput({UPred, geom}, divGradInput)
    divGradInput = divGradInput[1]  -- just U component (ignore geom deriv).
  end

  -- Now set the input gradients.
  self.gradInput[1]:resizeAs(pPred)
  self.gradInput[2]:resizeAs(UPred)

  if self.pLambda > 0 then
    self.gradInput[1]:copy(pGradInput)
  else
    self.gradInput[1]:fill(0)
  end
  self.gradInput[2]:fill(0)
  if self.uLambda > 0 then
    self.gradInput[2]:add(uGradInput)
  end
  if self.divLambda > 0 then
    self.gradInput[2]:add(divGradInput)
  end

  return self.gradInput
end

function FluidCriterion:type(type)
  parent.type(self, type)
  self._pLoss:type(type)
  self._uLoss:type(type)
  self._divLoss:type(type)
  self._divNetwork:type(type)
  for key, value in pairs(self.gradInput) do
    self.gradInput[key] = value:type(type)
  end
  return self
end
