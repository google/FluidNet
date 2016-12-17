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

function torch.plotWeightHistograms(conf, mconf, model, data, criterion,
                                    verbose)
  if verbose == nil then
    verbose = false
  end

  -- Increase the batch size.
  local oldBatchSize = conf.batchSize
  conf.batchSize = 32

  -- Pick a random batch.
  model:clearState()
  collectgarbage()

  local imgList = torch.randperm(data:nsamples())[{{1, conf.batchSize}}]
  imgList = imgList:totable()
  local batchCPU, batchGPU = data:AllocateBatchMemory(conf.batchSize)
  local perturb = false
  data:CreateBatch(batchCPU, torch.IntTensor(imgList), conf.batchSize, perturb,
                   {}, mconf.netDownsample, conf.dataDir)

  -- FPROP AND BPROP a sample:
  local input = torch.getModelInput(batchGPU)
  local target = torch.getModelTarget(batchGPU)
  model:zeroGradParameters()
  local output = model:forward(input)
  local err = criterion:forward(output, target)
  local df_do = criterion:backward(output, target)
  model:backward(input, df_do)

  local wStd = {}
  local wMax = {}
  local outputStd = {}
  local outputMax = {}
  local gradInputStd = {}
  local gradInputMax = {}

  local convLayerId = 0
  for i = 1, #model.modules do
    local layer = model.modules[i]
    if torch.type(layer) == 'cudnn.SpatialConvolution' then
      collectgarbage()
      local conv = layer
      convLayerId = convLayerId + 1
      local w = conv.weight
      w = w:view(w:size(1) * w:size(2), 1, w:size(3), w:size(4))
      local nfilt = w:size(1)
      local nrow = math.ceil(math.sqrt(nfilt))
      local zoom = 100 / nrow
      local scaleeach = false
      if w:size(3) > 1 or w:size(4) > 1 then
        scaleeach = true
      end
      local legend = 'Layer ' .. convLayerId

      if verbose then
        image.display{image = w, padding = 1, zoom = zoom, nrow = nrow,
                      legend = legend .. ' weights', scaleEach = scaleeach}
      end
  
      -- Plot the weight histogram.
      w = w:view(w:numel())
      if verbose then
        gnuplot.figure()
        gnuplot.hist(w, 40)
        gnuplot.title(legend .. ' weight hist')
      end

      -- Plot the activation histogram.
      local outputCPU = layer.output:float():view(layer.output:numel())
      if verbose then
        gnuplot.figure()
        gnuplot.hist(outputCPU, 40)
        gnuplot.title(legend .. ' output hist')
      end

      -- Plot the gradient histogram.
      local gradCPU = layer.gradInput:float():view(layer.gradInput:numel())
      if verbose then
        gnuplot.figure()
        gnuplot.hist(gradCPU, 40)
        gnuplot.title(legend .. ' gradInput hist')
      end

      wStd[#wStd + 1] = w:std()
      outputStd[#outputStd + 1] = outputCPU:std()
      gradInputStd[#gradInputStd + 1] = gradCPU:std()
      wMax[#wMax + 1] = w:clone():abs():max()
      outputMax[#outputMax + 1] = outputCPU:clone():abs():max()
      gradInputMax[#gradInputMax + 1] = gradCPU:clone():abs():max()
    end
  end

  local function nicePlot(values, ylabel)
    local y = torch.FloatTensor(values)
    local x = torch.FloatTensor(torch.range(1, y:size(1)))
    gnuplot.figure()
    gnuplot.plot({ylabel, x, y, '-'})
    gnuplot.xlabel('Conv Layer')
    gnuplot.grid(true)
    gnuplot.ylabel(ylabel)
  end

  nicePlot(wStd, 'std(layer.weight)')
  nicePlot(outputStd, 'std(layer.output)')
  nicePlot(gradInputStd, 'std(layer.gradInput)')
  nicePlot(wMax, 'max(abs(layer.weight))')
  nicePlot(outputMax, 'max(abs(layer.output))')
  nicePlot(gradInputMax, 'max(abs(layer.gradInput))')

  -- Put back the old batch size.
  conf.batchSize = oldBatchSize
end
