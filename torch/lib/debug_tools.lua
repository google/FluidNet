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

-- Kris's collection of debug tools.
--
-- TODO(kris): This needs cleaning up (for style mostly).

function showFiltersOfType(model, type, prefix, monochrome)
    prefix = prefix or ''
    monochrome = monochrome or true
    
    local conv_nodes = model:findModules(type)
    for i = 1, #conv_nodes do
      local m1=conv_nodes[i]
      --TODO: how can I automatically tell the difference between monochrome and color filters???
      if not monochrome then
        image.display{m1.weight:view(m1.nOutputPlane, m1.nInputPlane, m1.kH, m1.kW), zoom = 10, legend=(prefix .. "conv layer w " .. i), padding = 1}
        image.display{m1.gradWeight:view(m1.nOutputPlane, m1.nInputPlane, m1.kH, m1.kW), zoom = 10, legend=(prefix .. "conv layer g " .. i), padding = 1}
      else
        local filterCount = m1.nOutputPlane*m1.nInputPlane
        local dim = filterCount
        if dim  > 64 then dim  = 64 end

        local w = m1.weight
        w = torch.reshape(w,filterCount*m1.kH*m1.kW) -- make it one dimensional
        w=w[{{1,dim*m1.kH*m1.kW}}]

        local g = m1.gradWeight
        g = torch.reshape(g,filterCount*m1.kH*m1.kW) -- make it one dimensional
        g=g[{{1,dim*m1.kH*m1.kW}}]
        image.display{image = w:view(dim, m1.kH, m1.kW), zoom = 10, legend=(prefix .. "conv layer  w " .. i), padding = 1}
        image.display{image = g:view(dim, m1.kH, m1.kW), zoom = 10, legend=(prefix .. "conv layer g " .. i), padding = 1}
      end
    end
end

function saveFiltersToImage(model, type, prefix, monochrome, maxFilters, conf, mconf)
    prefix = prefix or ''
    monochrome = monochrome or true
    maxFilters = maxFilters or 16
    
    local conv_nodes = model:findModules(type)
    for i = 1, #conv_nodes do
      local m1=conv_nodes[i]
      --TODO: how can I automatically tell the difference between monochrome and color filters???
      if not monochrome then
        image.display{m1.weight:view(m1.nOutputPlane, m1.nInputPlane, m1.kH, m1.kW), zoom = 10, legend=(prefix .. "conv layer w " .. i), padding = 1}
        image.display{m1.gradWeight:view(m1.nOutputPlane, m1.nInputPlane, m1.kH, m1.kW), zoom = 10, legend=(prefix .. "conv layer g " .. i), padding = 1}
      else
        if(m1.kH >= 5) then --only write large kernels to disk
          local filterCount = m1.nOutputPlane*m1.nInputPlane
          local dim = filterCount
          --Don't write all of the filters to disk --
          if dim  > maxFilters then dim  = maxFilters end

          local w = m1.weight
          w = torch.reshape(w,filterCount*m1.kH*m1.kW) -- make it one dimensional
          w=w[{{1,dim*m1.kH*m1.kW}}]--restrict it to n=dim images of size (kHxkW) n x kH x kW

          local g = m1.gradWeight
          g = torch.reshape(g,filterCount*m1.kH*m1.kW) -- make it one dimensional
          g=g[{{1,dim*m1.kH*m1.kW}}]--restrict it to n=dim images of size (kHxkW)

          w = w:view(dim, m1.kH, m1.kW) 
          w = image.toDisplayTensor{input = w, padding = 1, nrow = math.sqrt(dim)}
          g = g:view(dim, m1.kH, m1.kW)
          g = image.toDisplayTensor{input = g, padding = 1, nrow = math.sqrt(dim)}
          local filename = conf.imageDir .."gpu_" .. conf.gpu .. "_epoch_" .. mconf.epoch .. prefix 
          image.save(filename .. "_convlayer_weights_" .. i .. ".png",  w)
          image.save(filename .. "_convlayer_gradients_" .. i .. ".png", g)
        end
      end
    end
end

function displayWeightsAndGradients(model, showGradients)
    --The following get all the weights as a table.  You can then iterate through them
    --but you would have to make sure that there are two dimensions to each tensor
    --if there aren't then you need to repeatTensor it
    showGradients = showGradients or false
    local weights2D, gradWeights2D = model:parameters() -- this returns unflattened tensors getParameters would return flattened tensors
    for i = 1, #weights2D do
      if weights2D[i]:nDimension() > 1 then
        image.display{image = weights2D[i], legend=('_model weights layer ' .. i)}
        if showGradients then
          image.display{image = gradWeights2D[i], legend=('_model gradweights layer ' .. i)}
        end
      else
        image.display{image = weights2D[i]:repeatTensor(2,1), legend=('_model weights layer ' .. i)}
        if showGradients then
          image.display{image = gradWeights2D[i]:repeatTensor(2,1), legend=('_model gradweights layer ' .. i)}
        end
      end
    end
end

function locals()
  local variables = {}
  local idx = 1
  while true do
    local ln, lv = debug.getlocal(2, idx)--level of stack is parent... 1 would be local to this scope
    if ln ~= nil then
      variables[ln] = lv
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

function local_names()
  local variables = {}
  local idx = 1
  while true do
    local ln, lv = debug.getlocal(2, idx)
    if ln ~= nil then
      variables[ln] = ln
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

function upvalues()
  local variables = {}
  local idx = 1
  local func = debug.getinfo(2, "f").func
  while true do
    local ln, lv = debug.getupvalue(func, idx)
    if ln ~= nil then
      variables[ln] = lv
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

function upvalue_names()
  local variables = {}
  local idx = 1
  local func = debug.getinfo(2, "f").func
  while true do
    local ln, lv = debug.getupvalue(func, idx)
    if ln ~= nil then
      variables[ln] = ln
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end
