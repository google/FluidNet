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

-- Recursive function to calculate total flops and peak memory usage.
-- For memory we assume we need input & output tensors (for a single layer,
-- because we will ping-pong buffers) plus whatever the layer needs (weights,
-- biases, etc).
-- Only a subset of nn modules are supported.
-- NOTE: this is approximate! It is just an estimate per layer from the
-- analytic function each layer describes. It is NOT an actual measure of the
-- flops in the torch implementation or as would be implemented by any specific
-- DSP.

-- IMPORTANT *** This function assumes that module:forward has been called ***.

-- @param module: nn module
-- @param depth: (optional) leave as nil for top-level call.
-- @param verbose: (optional) verbosity level (0, 1, 2)
function torch.CalculateFlops(module, input, verbose, depth)
  local depth = depth or 0
  verbose = verbose or 0
  local moduleType = torch.type(module)
  local flops, peakMemory

  local function calcSingleBatchNumel(tensor)
    assert(torch.isTensor(tensor))
    assert(tensor:dim() > 1, 'No batch dim on input tensor')
    local numel = 1
    for i = 2, tensor:dim() do
      numel = numel * tensor:size(i)
    end
    return numel
  end

  local spaces = ''
  for i = 1, depth do
    spaces = spaces .. '  '
  end


  local outputNumel, inputNumel
  if torch.isTensor(module.output) then
    outputNumel = calcSingleBatchNumel(module.output)
  end
  if torch.isTensor(input) then
    inputNumel = calcSingleBatchNumel(input)
  end

  if moduleType == 'nn.Sequential' or moduleType == 'nn.ParallelTable' or
     moduleType == 'nn.Concat' or moduleType == 'nn.ConcatTable' then
    -- Recurse on sub-modules. Add the flops, but take the max of peak memory.
    flops = 0
    peakMemory = 0
    for i = 1, #module.modules do
      local curFlops, curPeakMemory
      if moduleType == 'nn.Sequential' then
        curFlops, curPeakMemory =
            torch.CalculateFlops(module.modules[i], input, verbose, depth + 1)
        input = module.modules[i].output
      elseif moduleType == 'nn.ConcatTable' then
        curFlops, curPeakMemory =
             torch.CalculateFlops(module.modules[i], input, verbose, depth + 1)
      else  -- ParallelTable, Concat
        assert(torch.type(input) == 'table' and #input == module.modules)
        print(module)
        print(i)
        curFlops, curPeakMemory =
            torch.CalculateFlops(module.modules[i], input[i], verbose,
                                 depth + 1)
      end
      flops = flops + curFlops
      peakMemory = math.max(peakMemory, curPeakMemory)
    end
  elseif moduleType == 'nn.ReLU' or moduleType == 'cudnn.ReLU' then
    -- ReLU = 1 flop per element (max(x, y))
    flops = outputNumel
    peakMemory = outputNumel  -- We can do relu-inplace.
  elseif moduleType == 'nn.Sigmoid' then
    -- f_i(x) = log(1 / (1 + exp(-x_i)))  --> Guess ~4 flops (probably wrong)
    flops = outputNumel * 4
    peakMemory = outputNumel  -- We can do relu-inplace.
  elseif moduleType == 'nn.Padding' then
    flops = 0
    -- Worst case, we may have to keep both input and output data based on
    -- how the data is stored and what dimensions we need to pad.
    peakMemory = outputNumel + inputNumel
  elseif moduleType == 'nn.SpatialConvolutionMM' or
         moduleType == 'cudnn.SpatialConvolution' or
         moduleType == 'nn.SpatialConvolution' or
         moduleType == 'nn.SpatialConvolutionUpsample' then
    if moduleType == 'nn.SpatialConvolutionUpsample' then
      module = module.modules[1]
      assert(torch.type(module) == 'nn.SpatialConvolution' or
             torch.type(module) == 'cudnn.SpatialConvolution')
    end
    -- From Torch7 docs:
    -- The output value of the layer can be precisely described as:
    --   output[i][j][k] = bias[k]
    --     + sum_l sum_{s=1}^kW sum_{t=1}^kH weight[s][t][l][k]
    --                                       * input[dW*(i-1)+s)][dH*(j-1)+t][l]
    -- It's easier to see in this image:
    -- http://deeplearning.net/tutorial/_images/cnn_explained.png
    local inFeats = module.nInputPlane
    -- (1 +) is for bias
    -- (* 2) is one flop for weight-pixel multiplication and one accumulation
    local flopsPerOutput = 1 + inFeats * module.kW * module.kH * 2
    flops = flopsPerOutput * outputNumel
    peakMemory = outputNumel + inputNumel +
                 module.bias:numel() + module.weight:numel()
  elseif moduleType == 'nn.VolumetricConvolution' or
         moduleType == 'cudnn.VolumetricConvolution' or
         moduleType == 'nn.VolumetricConvolutionUpsample' then
    if moduleType == 'nn.VolumetricConvolutionUpsample' then
      module = module.modules[1]
      assert(torch.type(module) == 'nn.VolumetricConvolution' or
             torch.type(module) == 'cudnn.VolumetricConvolution')
    end
    local inFeats = module.nInputPlane
    local flopsPerOutput = 1 + inFeats * module.kT * module.kW * module.kH * 2
    flops = flopsPerOutput * outputNumel
    peakMemory = outputNumel + inputNumel +
                 module.bias:numel() + module.weight:numel()
  elseif moduleType == 'nn.SpatialMaxPooling' or
         moduleType == 'cudnn.SpatialMaxPooling' then
    -- Assume one flop per max (like with relu). For a 2x2 pooling = 4x
    -- decimation, there are n-1 max calls, so 3 flops per output pixel.
    local receptiveFieldPixels = module.kW * module.kH
    flops = (receptiveFieldPixels - 1) * outputNumel
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'nn.VolumetricMaxPooling' or
         moduleType == 'cudnn.VolumetricMaxPooling' then
    local receptiveFieldPixels = module.kT * module.kW * module.kH
    flops = (receptiveFieldPixels - 1) * outputNumel
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'nn.CAddTable' then
    -- Flops is equal to the (number of input tables - 1) * numel, i.e.:
    -- Y = A + B + C + ... + N --> n-1 additions per pixel.
    assert(torch.type(input) == 'table')
    flops = (#input - 1) * outputNumel
    peakMemory = outputNumel + #input * outputNumel
  elseif moduleType == 'nn.CSubTable' then
    -- Flops is equal to the size of the output (i.e. just A - B).
    flops = outputNumel
    peakMemory = outputNumel * 3
  elseif moduleType == 'nn.SpatialUpSamplingNearest' then
    -- This layer is all memory io and so doesn't use any flops.
    flops = 0
    peakMemory = outputNumel
  elseif moduleType == 'nn.SelectTable' or moduleType == 'nn.Identity' or
         moduleType == 'nn.SpatialDropout' or
         moduleType == 'nn.SpatialBatchNormalization' or
         moduleType == 'nn.Narrow' or moduleType == 'nn.JoinTable' or
         moduleType == 'nn.View' or moduleType == 'nn.BatchNormalization' or
         moduleType == 'nn.Select' or moduleType == 'nn.Unsqueeze' then
    -- These are layers that wouldn't actually exist in the embedded
    -- implementation (i.e. it's a memory reference), or they could be folded
    -- into the preceding convolution layer (as with dropout and normalization).
    -- DON'T COUNT THEM.
    flops = 0
    peakMemory = 0
  elseif moduleType == 'nn.SpatialContrastiveNormalizationBatch' then
    flops, peakMemory = torch.CalculateFlops(module.modules[1], input,
                                             verbose, depth + 1)
  elseif moduleType == 'nn.SpatialSubtractiveNormalizationBatch' then
    -- It's a local averaging (across all input features), followed by one
    -- subtractive step.
    assert(module.kernel:dim() == 2, 'Expecting 2D normalization kernel')
    assert(input:dim() == 4, 'Expecting 4D input')
    local inFeats = module.nInputPlane
    -- (* 2) is one flop for weight multiplication and one accumulation
    local flopsPerMeanPixel =
      inFeats * module.kernel:size(1) * module.kernel:size(2) * 2
    local inWidth = input:size(4)
    local inHeight = input:size(3)
    local flopsPerMean = flopsPerMeanPixel * inWidth * inHeight
    flops = flopsPerMean + inputNumel
    peakMemory = inputNumel + inWidth * inHeight + outputNumel
  elseif moduleType == 'nn.SpatialDivisiveNormalizationBatch' then
    assert(module.kernel:dim() == 2, 'Expecting 2D normalization kernel')
    assert(input:dim() == 4, 'Expecting 4D input')
    -- We need to calculate the var, which is calculating sum(x^2) and sum(x)^2
    -- Again, it is one variance per input pixel stack.
    local inFeats = module.nInputPlane
    -- (* 4) is one flop for x kernel multiplication, one for x kernel
    -- accumulation, one for x^2 calculation, one for x^2 accumulation.
    local flopsPerVarPixel =
      inFeats * module.kernel:size(1) * module.kernel:size(2) * 4
    local inWidth = input:size(4)
    local inHeight = input:size(3)
    local flopsPerVar = flopsPerVarPixel * inWidth * inHeight
    flops = flopsPerVar + inputNumel
    peakMemory = inputNumel + 2 * inWidth * inHeight + outputNumel
  elseif moduleType == 'nn.Linear' then
    assert(input:dim() == 2, 'Expecting 2D input')
    -- Applies y = Ax + b
    assert(module.weight:size(2) == inputNumel)
    assert(module.weight:size(1) == outputNumel)
    -- mat-vec mult/add + bias.
    flops = 2 * inputNumel * outputNumel + outputNumel
    peakMemory = inputNumel + outputNumel + module.weight:numel()
  elseif moduleType == 'nn.gModule' then
    -- Walk the forward nodes.
    -- This assumes the graph is static I think (i.e. we don't
    -- regenerate module.forwardnodes = module.fg:topsort().
    flops = 0
    peakMemory = 0
    for _, child in pairs(module.forwardnodes) do
      assert(torch.type(child) == 'nngraph.Node')
      assert(child.data.input ~= nil)
      local childFlops, childPeakMemory =
          torch.CalculateFlops(child, child.data.input, verbose, depth + 1)
      flops = flops + childFlops
      peakMemory = math.max(peakMemory, childPeakMemory)
    end
  elseif moduleType == 'nngraph.Node' then
    -- Add up the FLOPS of the current node's module (if it exists).
    flops = 0
    peakMemory = 0
    if module.data.module ~= nil then
      assert(module.data.input ~= nil)
      local curInput = module.data.input
      local curModule = module.data.module
      if #curInput == 1 then
        curInput = curInput[1]
      end
      flops, peakMemory =
          torch.CalculateFlops(curModule, curInput, verbose, depth + 1)
    end

    local debugLabel = '<empty debug label>'
    if module.data.annotations._debugLabel ~= nil then
      debugLabel = module.data.annotations._debugLabel
    end
    local name = '<unnamed node>'
    if module.data.annotations.name ~= nil then
      name = module.data.annotations.name
    end
    if verbose > 1 then
      print(spaces .. 'internal node flops (name: ' .. name ..
            ', debugLabel: ' .. debugLabel .. '): ' .. flops)
     end
    -- We wont recurse the children. Instead when we encounter an nn.gModule
    -- we use the forwardNodes table to visit all nodes in the graph.
  elseif moduleType == 'tfluids.VelocityUpdate' then
    -- This is VERY rough (from a cursory glance at tfluids/generic/tfluids.cc).
    local pNumel = calcSingleBatchNumel(input[1])
    local flagsNumel = calcSingleBatchNumel(input[3])
    local UNumel = calcSingleBatchNumel(input[2])
    flops = flagsNumel + 2 * pNumel + UNumel
    peakMemory = UNumel + flagsNumel + pNumel  -- Done in-place.
  elseif moduleType == 'tfluids.SetWallBcs' then
    -- This is VERY rough (from a cursory glance at tfluids/generic/tfluids.cc).
    local flagsNumel = calcSingleBatchNumel(input[2])
    local UNumel = calcSingleBatchNumel(input[1])
    flops = flagsNumel * 2
    peakMemory = flagsNumel + UNumel  -- Done in-place.
  elseif moduleType == 'tfluids.VelocityDivergence' then
    local flagsNumel = calcSingleBatchNumel(input[2])
    local UNumel = calcSingleBatchNumel(input[1])
    -- This is also VERY rough.
    flops = flagsNumel * 8
    peakMemory = flagsNumel * 2 + UNumel
  elseif moduleType == 'nn.StandardDeviation' then
    print('WARNING: Not estimating flops for nn.StandardDeviation')
    flops = 0
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'nn.Clamp' then
    flops = 2 * inputNumel
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'nn.ApplyScale' then
    inputNumel = calcSingleBatchNumel(input[1])
    outputNumel = inputNumel
    flops = 2 * inputNumel
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'tfluids.FlagsToOccupancy' then
    flops = 0
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'nn.Mul' then
    flops = 1 * inputNumel
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'nn.VolumetricAveragePooling' then
    flops = outputNumel * module.kW * module.kH * module.kT
    peakMemory = inputNumel + outputNumel
  elseif moduleType == 'tfluids.VolumetricUpSamplingNearest' then
    flops = 0  -- Just a copy.
    peakMemory = inputNumel + outputNumel
  else
    error('Module type ' .. moduleType .. ' is not supported')
  end

  local bytesToMB = 1 / math.pow(1024, 2)
  if verbose > 0 then
    if moduleType ~= 'nngraph.Node' or verbose > 1 then
      local moduleStr
      if module.__tostring ~= nil then
        moduleStr = module:__tostring()
      else
        moduleStr = moduleType
      end
      print(spaces .. torch.HumanReadableNumber(flops) .. ' flops: ' .. 
            moduleStr)
      print(spaces .. '  --> ' .. 
            string.format('%.2fMB', peakMemory * bytesToMB) .. ' peak bytes')
      if torch.type(input) == 'table' then
        print(spaces .. '  --> Input: ' ..
              torch.SerializeTable(input, '', true, true))
      elseif torch.isTensor(input) then
        print(spaces .. '  --> Input: ' .. torch.TensorToString(input))
      end
    end
  end

  return flops, peakMemory
end

