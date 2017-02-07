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

-- dofile('lib/load_package_safe.lua')

local image = require('image')
torch.loadPackageSafe('qt')
torch.loadPackageSafe('qttorch')
torch.loadPackageSafe('qtwidget')
torch.loadPackageSafe('qtuiloader')
local nn = torch.loadPackageSafe('nn')
local ffi = require('ffi')
local bit = require('bit')
local xlua = require('xlua')
torch.loadPackageSafe('cutorch')

-- This is a file of random code snippits.  It should probably be refactored
-- at some point, but it's where I dump my shared helper functions that don't
-- fit in anywhere else.

-- normalizeImageScale will crop and scale an image so that it is centered in
-- the destination image.  The scale and padding are returned to the user.
function torch.normalizeImageScale(im_src, im_dst)
  local dst_w = im_dst:size(3)
  local dst_h = im_dst:size(2)
  local src_w = im_src:size(3)
  local src_h = im_src:size(2)
  im_dst:fill(0)

  local scale = math.min(dst_w / src_w, dst_h / src_h)

  local w = torch.round(src_w * scale)
  local h = torch.round(src_h * scale)
  -- We can't EXACTLY maintain the same aspect ratio, be we will be as close as
  -- we can (maybe 0.5 pixel off)
  local im = image.scale(im_src, w, h, 'bilinear')

  -- We're not quite done, because we also need to center the image
  local padx = math.floor((dst_w - w) / 2)
  local pady = math.floor((dst_h - h) / 2)

  local range_y = {1 + pady, 1 + pady + h - 1}
  local range_x = {1 + padx, 1 + padx + w - 1}

  im_dst[{{}, range_y, range_x}]:copy(im)

  return scale, padx, pady
end

--[[
-- A quick and dirty test for the above
im_src = image.scale(image.lena(), 128, 128)
-- CASE 1: Downsample, limited in x
im_dst = torch.FloatTensor(3, 32, 64)
print(torch.normalizeImageScale(im_src, im_dst))  -- Expect 0.25, 32, 0
image.display{image=im_dst, zoom=8}
-- CASE 2: Downsample, limited in y
im_dst = torch.FloatTensor(3, 64, 32)
print(torch.normalizeImageScale(im_src, im_dst))  -- Expect 0.25, 0, 32
image.display{image=im_dst, zoom=8}
-- CASE 3: Upsample, limited in x
im_dst = torch.FloatTensor(3, 256, 512)
print(torch.normalizeImageScale(im_src, im_dst))  -- Expect 0.25, 128, 0
image.display{image=im_dst, zoom=1}
-- CASE 4: Upsample, limited in y
im_dst = torch.FloatTensor(3, 512, 256)
print(torch.normalizeImageScale(im_src, im_dst))  -- Expect 0.25, 0, 128
image.display{image=im_dst, zoom=1}
--]]

-- addBorder: Add a border to the image, where the pixel values are clamped
function torch.addBorder(im, border_size)
  assert(im:dim() == 2, 'input must be dimension 2')
  local ret_sz = {im:size(1) + 2 * border_size, im:size(2) + 2 * border_size}
  local ret_im = im:clone():resize(unpack(ret_sz))

  local vbl = {1, border_size}  -- v range border left
  local v = {vbl[2] + 1, vbl[2] + 1 + im:size(im:dim() - 1) - 1}
  local vbr = {v[2] + 1, v[2] + 1 + border_size - 1}  -- v range border right

  local ubl = {1, border_size}
  local u = {ubl[2] + 1, ubl[2] + 1 + im:size(im:dim()) - 1}
  local ubr = {u[2] + 1, u[2] + 1 + border_size - 1}

  -- center
  ret_im[{v, u}]:copy(im)
  -- up
  ret_im[{vbl, u}]:copy(im[{{1}, {}}]:expandAs(ret_im[{vbl, u}]))
  -- down
  ret_im[{vbr, u}]:copy(im[{{im:size(1)}, {}}]:expandAs(ret_im[{vbr, u}]))
  -- left
  ret_im[{v, ubl}]:copy(im[{{}, {1}}]:expandAs(ret_im[{v, ubl}]))
  -- right
  ret_im[{v, ubr}]:copy(im[{{}, {im:size(2)}}]:expandAs(ret_im[{v, ubr}]))
  -- left + up
  ret_im[{vbl, ubl}] = im[1][1]
  -- right + up
  ret_im[{vbl, ubr}] = im[1][im:size(2)]
  -- left + down
  ret_im[{vbr, ubl}] = im[im:size(1)][1]
  -- right + down
  ret_im[{vbr, ubr}] = im[im:size(1)][im:size(2)]

  return ret_im
end

-- addBorder: Add a border to the right of the image, where the pixel values are
-- clamped
function torch.addBorderBottomAndRight(im, border_size)
  assert(im:dim() == 2, 'input must be dimension 2')
  local ret_sz = {im:size(1) + border_size, im:size(2) + border_size}
  local ret_im = im:clone():resize(unpack(ret_sz))

  local v = {1, im:size(im:dim() - 1)}  -- v range img
  local vbr = {v[2] + 1, v[2] + 1 + border_size - 1}  -- v range border right

  local u = {1, im:size(im:dim())}
  local ubr = {u[2] + 1, u[2] + 1 + border_size - 1}

  -- center
  ret_im[{v, u}]:copy(im)
  -- down
  ret_im[{vbr, u}]:copy(im[{{im:size(1)}, {}}]:expandAs(ret_im[{vbr, u}]))
  -- right
  ret_im[{v, ubr}]:copy(im[{{}, {im:size(2)}}]:expandAs(ret_im[{v, ubr}]))
  -- right + down
  ret_im[{vbr, ubr}] = im[im:size(1)][im:size(2)]

  return ret_im

end

function torch.addConstantBorder(im, border_size, border_value)
  assert(im:dim() == 2, 'input must be dimension 2')
  local ret_sz = {im:size(1) + 2 * border_size, im:size(2) + 2 * border_size}
  local ret_im = im:clone():resize(unpack(ret_sz))

  local v = {border_size + 1, border_size + 1 + im:size(im:dim() - 1) - 1}
  local u = {border_size + 1, border_size + 1 + im:size(im:dim()) - 1}

  -- center
  ret_im:fill(border_value)
  ret_im[{v, u}]:copy(im)

  return ret_im
end

function torch.padImage(im, pad_lrtb, border_value)
  assert(im:dim() == 3 or im:dim() == 2, 'input must be dimension 2 or 3')
  assert(#pad_lrtb == 4, 'Need 4 pad values')
  for i = 1, #pad_lrtb do
    assert(pad_lrtb[i] >= 0, 'Padding must be >= 0')
  end
  local ret_sz
  if im:dim() == 2 then
    ret_sz = {im:size(1) + pad_lrtb[1] + pad_lrtb[2],
      im:size(2) + pad_lrtb[3] + pad_lrtb[4]}
  else
    ret_sz = {im:size(1), im:size(2) + pad_lrtb[3] + pad_lrtb[4],
      im:size(3) + pad_lrtb[1] + pad_lrtb[2]}
  end
  local ret_im = im:clone():resize(unpack(ret_sz))

  -- v range img
  local v = {pad_lrtb[3] + 1, pad_lrtb[3] + 1 + im:size(im:dim() - 1) - 1}
  local u = {pad_lrtb[1] + 1, pad_lrtb[1] + 1 + im:size(im:dim()) - 1}

  -- center
  ret_im:fill(border_value)
  if im:dim() == 2 then
    ret_im[{v, u}]:copy(im)
  else
    ret_im[{{}, v, u}]:copy(im)
  end

  return ret_im
end

-- NOTE: This doesn't handle the case where the patch goes off the image (but
-- it will throw a warning)
function torch.cropImage(img, xy, patch_size, patch_center)
  local range_u = {xy[1] - patch_center + 1, xy[1] + patch_size - patch_center}
  local range_v = {xy[2] - patch_center + 1, xy[2] + patch_size - patch_center}
  if img:dim() == 2 then
    assert(range_u[1] >= 1 and range_u[2] <= img:size(2), 'patch is off image')
    assert(range_v[1] >= 1 and range_v[2] <= img:size(1), 'patch is off image')
    return img[{range_v, range_u}]
  elseif img:dim() == 3 then
    assert(range_u[1] >= 1 and range_u[2] <= img:size(3), 'patch is off image')
    assert(range_v[1] >= 1 and range_v[2] <= img:size(2), 'patch is off image')
    return img[{{}, range_v, range_u}]
  else
    error('img must be 2D or 3D')
  end
end


-- Note: imA and imB will be modified!
-- This performs per-pixel alpha blending from here:
-- http://en.wikipedia.org/wiki/Alpha_compositing
function torch.alphaBlend(imA, imB)
  local ret = imA:clone()
  assert(imA:size(1) == 4 and imB:size(1) == 4, 'You need an alpha chan')
  assert(imA:dim() == 3 and imB:dim() == 3, 'You need a 3D tensor')
  local tmp = torch.Tensor():typeAs(imA):resize(1, imA:size(2), imA:size(3))

  -- Calculate the output RGB chans
  -- C_a * alpha_a
  imA[{{1, 3}, {}, {}}]:cmul(imA[{{4}, {}, {}}]:expandAs(imA[{{1, 3}, {}, {}}]))
  -- C_b * alpha_b
  imB[{{1, 3}, {}, {}}]:cmul(imB[{{4}, {}, {}}]:expandAs(imB[{{1, 3}, {}, {}}]))
  tmp:copy(imA[{{4}, {}, {}}]):mul(-1):add(1)  -- (1 - alpha_a)
  -- C_b * alpha_b * (1 - alpha_a)
  imB[{{1, 3}, {}, {}}]:cmul(tmp:expandAs(imB[{{1, 3}, {}, {},}]))

  -- C_a * alpha_a + C_b * alpha_b * (1 - alpha_a)
  ret[{{1, 3}, {}, {}}]:copy(imA[{{1, 3}, {}, {}}]):add(imB[{{1, 3}, {}, {}}])

  -- Calculate the ouptut alpha chan
  ret[4]:copy(imB[4]):cmul(tmp)  -- alpha_b * (1 - alpha_a)
  ret[4]:add(imA[4])  -- alpha_a + alpha_b * (1 - alpha_a)

  return ret
end

function torch.sizeAsTable(x)
  local return_sizes = {}
  if type(x) == 'table' then
    for i = 1, #x do
      return_sizes[i] = torch.sizeAsTable(x[i])
    end
  else
    assert(torch.typename(x) == 'torch.FloatTensor' or torch.typename(x) ==
      'torch.CudaTensor', 'torch.sizeAsTable incorrect input')
    for i = 1, x:dim() do
      return_sizes[i] = x:size(i)
    end
  end
  return return_sizes
end

function torch.typeAsTable(x)
  local return_types = {}
  if type(x) == 'table' then
    for i = 1, #x do
      return_types[i] = torch.typeAsTable(x[i])
    end
  else
    return_types = torch.typename(x)
  end
  return return_types
end

-- A very simple (and also slow) downsample function.  It is so that I can be
-- absolutely sure which pixels are sampled when using nearest neighbour samples
-- Input size should be 2D and is of size (height x width)
function torch.downsampleSimple(x, ratio)
  assert(x:dim() == 2, 'incorrect size input')
  local ret = torch.FloatTensor(x:size(1) / ratio, x:size(2) / ratio)
  for vdst = 1, ret:size(1) do
    local vsrc = (vdst - 1) * ratio + 1
    for udst = 1, ret:size(2) do
      local usrc = (udst - 1) * ratio + 1
      ret[{vdst, udst}] = x[{vsrc, usrc}]
    end
  end
  return ret
end
function torch.upsampleSimple(x, ratio)
  assert(x:dim() == 2, 'incorrect size input')
  local ret = torch.FloatTensor(x:size(1) * ratio, x:size(2) * ratio)
  for vsrc = 1, x:size(1) do
    for usrc = 1, x:size(2) do
      for voff = 0, ratio - 1 do
        for uoff = 0, ratio - 1 do
          local vdst = (vsrc - 1) * ratio + 1 + voff
          local udst = (usrc - 1) * ratio + 1 + uoff
          ret[{vdst, udst}] = x[{vsrc, usrc}]
        end
      end
    end
  end
  return ret
end
-- SOME Simple test code for the above
-- im = image.scale(image.rgb2y(image.lena()):squeeze(), 128, 128)
-- image.display(im)
-- image.display(torch.downsampleSimple(im, 8))
-- image.display(torch.upsampleSimple(im, 8))

-- clamp1DTensor takes in a tensor x and two tables 'low' and 'high' containing
-- the same number of elements as x.  It then clamps the values of x to the
-- corresponding values of low and high. ie:
-- uv = torch.rand(2) * 1000
-- torch.clamp1DTensor(uv, {1, 1}, {640, 480})
function torch.clamp1DTensor(x, low, high)
  assert(x:dim() == 1, 'input tensor is not 1D')
  assert(#low == x:size(1) and #high == x:size(1), 'inconsistent sizes')
  for i = 1, x:size(1) do
    x[i] = math.max(low[i], math.min(high[i], x[i]))
  end
end

-- setDeviceSafe - Avoid redundant calls to setDevice
function torch.setDeviceSafe(gpuid)
  if cutorch.getDevice() ~= gpuid then
    cutorch.setDevice(gpuid)
  end
end

function torch.zeroDataSize(data)
  if type(data) == 'table' then
    for i = 1, #data do
      data[i] = torch.zeroDataSize(data[i])
    end
  elseif type(data) == 'userdata' then
    data = torch.Tensor():typeAs(data)
  end
  return data
end

function torch.saveTensorToFile(filename, tensor)
  local out = torch.DiskFile(filename, 'w')
  out:binary()
  if torch.typename(tensor) == 'torch.FloatTensor' then
    out:writeFloat(tensor:storage())
  elseif torch.typename(tensor) == 'torch.DoubleTensor' then
    out:writeDouble(tensor:storage())
  elseif torch.typename(tensor) == 'torch.CudaTensor' then
    out:writeFloat(tensor:float():storage())
  else
    error('Tensor not supported by this function (but could be easily added)')
  end
  out:close()
end

function torch.loadTensorFromFile(filename, type)
  local fin = torch.DiskFile(filename, 'r')
  fin:binary()
  fin:seekEnd()
  local file_size_bytes = fin:position() - 1
  fin:seek(1)

  local tensor
  if type == 'float' then
    tensor = torch.FloatTensor(file_size_bytes / 4)
    fin:readFloat(tensor:storage())
  elseif type == 'double' then
    tensor = torch.DoubleTensor(file_size_bytes / 8)
    fin:readDouble(tensor:storage())
  elseif type == 'byte' then
    tensor = torch.ByteTensor(file_size_bytes)
    fin:readByte(tensor:storage())
  else
    error('Tensor not supported by this function (but could be easily added)')
  end
  fin:close()
  return tensor
end

function torch.getByteTensorSizeFromFile(filename)
  local fin = torch.DiskFile(filename, 'r')
  fin:binary()
  fin:seekEnd()
  local file_size_bytes = fin:position() - 1
  fin:close()
  return file_size_bytes
end


function torch.isNan(x)
  return x ~= x
end

function torch.copyTable(x)
  local x_ret = {}
  for key, value in pairs(x) do
    if type(value) == 'table' then
      x_ret[key] = torch.copyTable(value)  -- recursive call
    else
      if string.find(torch.type(value), 'Tensor') ~= nil then
        x_ret[key] = value:clone()
      else
        x_ret[key] = value
      end
    end
  end
  return x_ret
end

-- Note: this will create lots of dynamic data --> IT WILL BE SLOW
function torch.sigmoid(x)
  local tmp = torch.exp(x)
  local y = tmp:clone()
  return y:cdiv(tmp:add(1))
end

function torch.lastElem(T)
  return T[#T]
end

-- img_data is overwritten!  (it copies back into the tensor in place)
function torch.LCNImages(img_data, kernel_size, batch_size)
  kernel_size = kernel_size or 9
  batch_size = batch_size or 32

  assert(torch.typename(img_data) == 'torch.FloatTensor', 'bad input')
  assert(img_data:dim() == 4, 'bad input')
  local nimages = img_data:size(1)
  local nchans = img_data:size(2)
  local h = img_data:size(3)
  local w = img_data:size(4)

  local kernel = torch.Tensor(kernel_size, kernel_size):fill(1)
  local lcn = nn.SpatialContrastiveNormalizationBatch(nchans, kernel)

  local data = torch.FloatTensor(batch_size, nchans, h, w)
  for i = 1, nimages, batch_size do
    collectgarbage()
    -- Copy the image to the gpu and do the lcn there
    local in_range = {i, math.min(i + batch_size - 1, nimages)}
    local out_range = {in_range[1] - i + 1, in_range[2] - in_range[1] + 1}
    data[{out_range, {}, {}, {}}]:copy(img_data[{in_range, {}, {}, {}}])

    -- Perform the local contrast normalization
    local lcn = lcn:forward(data)

    -- Now get the image back from the GPU
    img_data[{in_range, {}, {}, {}}]:copy(lcn[{out_range, {}, {}, {}}])
  end
end

function torch.printModel(node, level)
  level = level or 1
  for i = 1, level do
    io.write('   ')
  end
  print(torch.typename(node))

  if node.modules ~= nil then
    if type(node.modules) == 'table' then
      for i = 1, #node.modules do
        local next_id = torch.printModel(node.modules[i], level + 1)
      end
    end
  end
end

function torch.bool2num(val)
  assert(type(val) == 'boolean', 'Input not boolean')
  return val and 1 or 0
end

-- Helper function to create a rotation matrix
function torch.rmat(mat, deg, s)
  local r = deg / 180 * math.pi
  mat[{1, 1}] = s * math.cos(r)
  mat[{1, 2}] = -s * math.sin(r)
  mat[{2, 1}] = s * math.sin(r)
  mat[{2, 2}] = s * math.cos(r)
end

-- Helper function to create a rotation matrix (2D affine)
function torch.rmat3(mat, deg, s)
  local r = deg / 180 * math.pi
  mat[{1, 1}] = s * math.cos(r)
  mat[{1, 2}] = -s * math.sin(r)
  mat[{1, 3}] = 0
  mat[{2, 1}] = s * math.sin(r)
  mat[{2, 2}] = s * math.cos(r)
  mat[{2, 3}] = 0
  mat[{3, 1}] = 0
  mat[{3, 2}] = 0
  mat[{3, 3}] = 1
end

-- Helper function to create a translation matrix (2D affine)
function torch.tmat3(mat, tx, ty)
  mat[{1, 1}] = 1
  mat[{1, 2}] = 0
  mat[{1, 3}] = ty
  mat[{2, 1}] = 0
  mat[{2, 2}] = 1
  mat[{2, 3}] = tx
  mat[{3, 1}] = 0
  mat[{3, 2}] = 0
  mat[{3, 3}] = 1
end

function torch.determinant2x2(M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 2 and M:size(2) == 2, "M is not 2x2")
  return M[{1, 1}] * M[{2, 2}] - M[{1, 2}] * M[{2, 1}]
end

function torch.determinant3x3(M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 3 and M:size(2) == 3, "M is not 3x3")
  return M[{1, 1}] * (M[{3, 3}] * M[{2, 2}] - M[{3, 2}] * M[{2, 3}]) -
         M[{2, 1}] * (M[{3, 3}] * M[{1, 2}] - M[{3, 2}] * M[{1, 3}]) +
         M[{3, 1}] * (M[{2, 3}] * M[{1, 2}] - M[{2, 2}] * M[{1, 3}])
end

function torch.inverse2x2(M_inv, M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 2 and M:size(2) == 2, "M is not 2x2")
  local det = torch.determinant2x2(M)
  if math.abs(det) < 1e-6 then
    error('Matrix is singular.  No torch.inverse exists.')
  end
  M_inv[{1, 1}] = M[{2, 2}]
  M_inv[{2, 2}] = M[{1, 1}]
  M_inv[{1, 2}] = -M[{1, 2}]
  M_inv[{2, 1}] = -M[{2, 1}]
  M_inv:mul(1 / det)
end

function torch.inverse3x3(M_inv, M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 3 and M:size(2) == 3, "M is not 3x3")
  local det = torch.determinant3x3(M)
  if math.abs(det) < 1e-6 then
    error('Matrix is singular.  No torch.inverse exists.')
  end
  M_inv[{1, 1}] = M[{3, 3}] * M[{2, 2}] - M[{3, 2}] * M[{2, 3}]
  M_inv[{1, 2}] = -(M[{3, 3}] * M[{1, 2}] - M[{3, 2}] * M[{1, 3}])
  M_inv[{1, 3}] = M[{2, 3}] * M[{1, 2}] - M[{2, 2}] * M[{1, 3}]
  M_inv[{2, 1}] = -(M[{3, 3}] * M[{2, 1}] - M[{3, 1}] * M[{2, 3}])
  M_inv[{2, 2}] = M[{3, 3}] * M[{1, 1}] - M[{3, 1}] * M[{1, 3}]
  M_inv[{2, 3}] = -(M[{2, 3}] * M[{1, 1}] - M[{2, 1}] * M[{1, 3}])
  M_inv[{3, 1}] = M[{3, 2}] * M[{2, 1}] - M[{3, 1}] * M[{2, 2}]
  M_inv[{3, 2}] = -(M[{3, 2}] * M[{1, 1}] - M[{3, 1}] * M[{1, 2}])
  M_inv[{3, 3}] = M[{2, 2}] * M[{1, 1}] - M[{2, 1}] * M[{1, 2}]
  M_inv:mul(1 / det)
end

do
  -- Test 2x2 torch.inverse
  local M = torch.rand(2, 2):double()
  local M_inv = M:clone():fill(0)
  torch.inverse2x2(M_inv, M)
  assert((torch.eye(2, 2):double() - torch.mm(M_inv, M)):abs():max() < 1e-6,
    'torch.inverse2x2 assert fails!')

  -- Test 3x3 torch.inverse
  M = torch.rand(3, 3):double()
  M_inv = M:clone():fill(0)
  torch.inverse3x3(M_inv, M)
  assert((torch.eye(3, 3):double() - torch.mm(M_inv, M)):abs():max() < 1e-6,
    'torch.inverse3x3 assert fails!')
end

function torch.symmetricFloor(x)
  if x < 0 then
    x = math.ceil(x)
  else
    x = math.floor(x)
  end
  return x
end

function torch.toboolean(str)
  if string.lower(str) == 'true' then
    return true
  elseif string.lower(str) == 'false' then
    return false
  else
    error('toboolean expects "true" or "false" inputs only')
  end
end

function torch.round(num)
  local under = math.floor(num)
  local upper = math.floor(num) + 1
  local underV = -(under - num)
  local upperV = upper - num
  if upperV > underV then
    return under
  else
    return upper
  end
end

function torch.range(start, stop)
  local ret = {}
  for i = start, stop do
    ret[#ret + 1] = i
  end
  return ret
end

-- This is slow.  Use spairingly.
function torch.inverseSoftPlus(y, beta)
  local x = y:clone()
  -- y = (1/k)*log(1+exp(k*x)) --> x = (1/k)*log(exp(k*y)-1)
  local mask = x:mul(beta):gt(y:clone():fill(20))
  local inverse_mask = mask:clone():mul(-1):add(1)  -- (1 - mask)
  x:exp():add(-1):add(1e-6):log():mul(1 / beta)
  x:cmul(inverse_mask:typeAs(x))
  local y_copy = y:clone():cmul(mask:typeAs(y))
  x:add(y_copy)
  return x
end

do
  if nn ~= nil then
     -- Test torch.inverseSoftPlus
     local sp = nn.SoftPlus(2):double()
     local output = torch.rand(5, 5):double()
     local err = output - sp:forward(torch.inverseSoftPlus(output, 2))
     assert(err:abs():max() < 1e-6, 'torch.inverseSoftPlus fails!')
  end
end

-- Unlike the one in the image package, this supports cuda tensors
-- NOTE: dx and dy are rounded (floor) to the nearest integer
function torch.translateImage(dst, src, dx, dy)
  assert(dst:isSameSizeAs(src), 'dst and src are not the same size!')
  local w, h
  local udim = dst:dim()
  local vdim = dst:dim() - 1
  w = dst:size(udim)
  h = dst:size(vdim)

  dx = math.floor(dx)
  dy = math.floor(dy)

  local udst =
      {math.min(math.max(1 + dx, 1), w), math.max(math.min(w, w + dx), 1)}
  local usrc =
      {math.min(math.max(1 - dx, 1), w), math.max(math.min(w, w - dx), 1)}
  local vdst =
      {math.min(math.max(1 + dy, 1), h), math.max(math.min(h, h + dy), 1)}
  local vsrc =
      {math.min(math.max(1 - dy, 1), h), math.max(math.min(h, h - dy), 1)}

  local dst_range = {}
  for i = 1, dst:dim() - 2 do
    dst_range[#dst_range + 1] = {}
  end
  dst_range[#dst_range + 1] = vdst
  dst_range[#dst_range + 1] = udst

  local src_range = {}
  for i = 1, src:dim() - 2 do
    src_range[#src_range + 1] = {}
  end
  src_range[#src_range + 1] = vsrc
  src_range[#src_range + 1] = usrc

  dst:fill(0)
  dst[dst_range]:copy(src[src_range])
end

--[[ MANUAL TEST FOR THE ABOVE
im = image.lena()
im_translate = im:clone()
torch.translateImage(im_translate, im, -5, -10)
image.display(im_translate)
--]]

function string:split(sep)
  assert(string.len(sep) == 1, 'separator must be a single char')
  local sep, fields = sep or ":", {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(c) fields[#fields + 1] = c end)
  return fields
end

function torch.FileExists(name)
   local f = io.open(name, "r")
   if f ~= nil then io.close(f) return true else return false end
end

function torch.readLinesFromFile(file)
  assert(torch.FileExists(file), 'file ' .. file .. ' does not exist!')
  local lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

torch.readFile = torch.readLinesFromFile  -- Alias

function torch.drawHorzLine(img, uv1, uv2, color)
  assert(uv1[2] == uv2[2], 'uv1 and uv2 arent horizontal')
  assert(img:dim() == 3, 'image is not of dimension 3')
  -- Clip the points to the image boundry
  uv1[1] = math.max(math.min(img:size(3), uv1[1]), 1)
  uv2[1] = math.max(math.min(img:size(3), uv2[1]), 1)
  local v = math.max(math.min(img:size(2), uv2[2]), 1)
  -- Draw the line
  for u = math.min(uv1[1], uv2[1]), math.max(uv1[1], uv2[1]) do
    img[{1, v, u}] = color[1]
    img[{2, v, u}] = color[2]
    img[{3, v, u}] = color[3]
  end
end

function torch.drawVertLine(img, uv1, uv2, color)
  assert(uv1[1] == uv2[1], 'uv1 and uv2 arent vertical')
  assert(img:dim() == 3, 'image is not of dimension 3')
  -- Clip the points to the image boundry
  uv1[2] = math.max(math.min(img:size(2), uv1[2]), 1)
  uv2[2] = math.max(math.min(img:size(2), uv2[2]), 1)
  local u = math.max(math.min(img:size(3), uv2[1]), 1)
  -- Draw the line
  for v = math.min(uv1[2], uv2[2]), math.max(uv1[2], uv2[2]) do
    img[{1, v, u}] = color[1]
    img[{2, v, u}] = color[2]
    img[{3, v, u}] = color[3]
  end
end

local colors = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}, {1, 0, 1},
    {0, 1, 1}, {0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0, 0.5},
    {0.75, 0.75, 0}, {0, 0.5, 0.5}, {0.5, 1, 0}, {0.5, 0, 1}, {0.5, 1, 1,},
    {1, 0.5, 0}, {0, 0.5, 1}, {1, 0.5, 1}, {1, 0, 0.5}, {0, 1, 0.5},
    {1, 1, 0.5}, {0.5, 0.5, 0}}
local color_names = {'red', 'green', 'blue', 'yellow', 'pink', 'cyan',
    'dark red', 'dark green', 'dark blue', 'purple', 'light olive', 'teal',
    'fluro green', 'light purple', 'light cyan', 'orange', 'dodger blue',
    'violet', 'deep pink', 'turquoise', 'khaki', 'olive'}
assert(#colors == #color_names, 'bad colors array')

function torch.drawLabels(img, uv_labels, w, h, start_color, sigma)
  start_color = start_color or 1
  local max_val = img:max()
  local min_val = img:min()
  local amp = math.max(max_val - min_val, 1e-3)
  local k, nlabels
  if uv_labels:dim() == 1 then
    nlabels = 1
  else
    nlabels = uv_labels:size(2)
  end

  for k = 1, nlabels do
    -- TODO: This will fail if all the images are not the same size
    local uv_pos
    if uv_labels:dim() == 1 then
      uv_pos = {uv_labels[1], uv_labels[2]}
    else
      uv_pos = {uv_labels[{1, k}], uv_labels[{2, k}]}
    end

    local color = {}
    local icolor = math.mod(k - 1 + start_color, #colors) + 1

    for c = 1, 3 do
      --table.insert(color, min_val + colors[icolor][c] * amp)
      table.insert(color, colors[icolor][c])
    end

    -- Draw a decaying gaussian centered at the pixel
    -- For consistancy, I'm calculating the gaussian in the same way that
    -- image.gaussian does, where uv_pos = 0 is the middle of the first pixel
    -- and uv_pos = 1 is the middle of the last pixel.
    local mean_horz = (uv_pos[1] - 0.5) / (w - 1)
    local mean_vert = (uv_pos[2] - 0.5) / (h - 1)

    sigma = sigma or 0.01

    local mean_v = mean_vert * h + 0.5;  -- line 995 in image.c
    local mean_u = mean_horz * w + 0.5;

    local over_sigmau = 1.0 / (sigma * w)  -- Precalculate
    local over_sigmav = 1.0 / (sigma * w)

    local u_min = math.min(math.max(math.floor(mean_u - 3 * sigma * w), 1), w)
    local u_max = math.min(math.max(math.floor(mean_u + 3 * sigma * w), 1), w)
    local v_min = math.min(math.max(math.floor(mean_v - 3 * sigma * h), 1), h)
    local v_max = math.min(math.max(math.floor(mean_v + 3 * sigma * h), 1), h)

    for v = v_min, v_max do
     for u = u_min, u_max do
        local du = (u - mean_u) * over_sigmau
        local dv = (v - mean_v) * over_sigmav
        local amp = math.exp(-((du * du * 0.5) + (dv * dv * 0.5)))

        img[{1, v, u}] = amp * color[1] + (1 - amp) * img[{1, v, u}]
        img[{2, v, u}] = amp * color[2] + (1 - amp) * img[{2, v, u}]
        img[{3, v, u}] = amp * color[3] + (1 - amp) * img[{3, v, u}]
      end
    end
  end
end

function torch.TensorToString(val)
  assert(torch.isTensor(val))
  str = '[' .. torch.type(val) .. ' of size '
  for i = 1, val:dim() do
    str = str .. val:size(i)
    if i < val:dim() then
      str = str .. 'x'
    end
  end
  str = str .. ']'
  return str
end

function torch.SerializeTable(val, name, skipnewlines, skipname, depth)
  skipnewlines = skipnewlines or false
  skipname = skipname or false
  depth = depth or 0

  local tmp
  if not skipnewlines then
    tmp = string.rep(" ", depth)
  else
    tmp = ""
  end

  if name and (not skipname) then tmp = tmp .. name .. " = " end

  if type(val) == "table" then
      tmp = tmp .. "{" .. (not skipnewlines and "\n" or "")

      -- We need to know if we've come across the last key. In a hash set we
      -- have no way of doing this without first flattening the key set to a
      -- vector (table with integer keys).
      local keys = {}
      for k, _ in pairs(val) do
        keys[#keys + 1] = k
      end

      for ikey = 1, #keys do
          k = keys[ikey]
          v = val[k]
          tmp = (tmp ..
                 torch.SerializeTable(v, k, skipnewlines, skipname, depth + 1))
          if ikey < #keys then
            tmp = tmp .. ','
          end
          if not skipnewlines then
            tmp = tmp .. "\n"
          elseif ikey < #keys then
            tmp = tmp .. " "
          end
      end
      tmp = tmp .. string.rep(" ", depth) .. "}"
  elseif type(val) == "number" then
      tmp = tmp .. tostring(val)
  elseif type(val) == "string" then
      tmp = tmp .. string.format("%q", val)
  elseif type(val) == "boolean" then
      tmp = tmp .. (val and "true" or "false")
  elseif torch.isTensor(val) then
      tmp = tmp .. torch.type(val)
  else
      tmp = tmp .. "\"[inserializeable datatype:" .. type(val) .. "]\""
  end

  return tmp
end

function torch.stringTableToCharTensor(str_table)
  assert(torch.type(str_table) == 'table')
  for i = 1, #str_table do
    assert(torch.type(str_table[i]) == 'string')
  end
  -- Get the maximum string length
  local max_str_len = 0
  for i = 1, #str_table do
    max_str_len = math.max(max_str_len, string.len(str_table[i]))
  end
  -- Allocate the output tensor
  local char_tensor = torch.CharTensor(#str_table, max_str_len + 1)  -- + null
  char_tensor:fill(0)
  -- Now copy the strings into the buffer
  for i = 1, #str_table do
    collectgarbage()
    local cur_char_tensor = torch.stringToCharTensor(str_table[i])
    char_tensor[{i, {1, cur_char_tensor:size(1)}}]:copy(cur_char_tensor)
  end
  return char_tensor
end

function torch.charTensorToStringTable(char_tensor)
  assert(char_tensor:dim() == 2, '2D tensor expected!')
  local str_table = {}
  for i = 1, char_tensor:size(1) do
    str_table[#str_table + 1] = torch.charTensorToString(char_tensor[i])
  end
  return str_table
end

function torch.charTensorToString(char_tensor)
  assert(char_tensor:dim() == 1, '1D tensor expected!')
  local ptr = torch.data(char_tensor)
  return ffi.string(ptr)
end

function torch.stringToCharTensor(str)
  local str_len = string.len(str)
  local char_tensor = torch.CharTensor(str_len + 1)  -- Null terminated
  local ptr = torch.data(char_tensor)
  ptr[str_len] = 0  -- Null terminate
  ffi.copy(ptr, str)
  return char_tensor
end

-- Reads the file into a 1D tensor
function torch.readFileToCharTensor(filename)
  local file = torch.DiskFile(filename, 'r')
  file:seekEnd()
  local file_size = file:position()
  file:seek(1)
  local data = file:readChar(file_size - 1)
  file:close()
  return torch.CharTensor(data)
end

function torch.writeCharTensorToFile(filename, data)
  local file = torch.DiskFile(filename, 'w')
  file:writeChar(data:storage())
  file:close()
end

function torch.mmul32(x1, x2) --multiplication with modulo2 semantics
  return tonumber(ffi.cast('uint32_t', ffi.cast('uint32_t', x1) *
      ffi.cast('uint32_t', x2)))
end

function torch.mmul64(x1, x2) --multiplication with modulo2 semantics
  return tonumber(ffi.cast('uint64_t', ffi.cast('uint64_t', x1) *
      ffi.cast('uint64_t', x2)))
end


-- http://www.isthe.com/chongo/tech/comp/fnv/
function torch.CalculateFNV32(char_tensor)
  local pstr = char_tensor:data()
  local const = 16777619
  local hash = 2166136261
  while pstr[0] ~= 0 do
    hash = bit.bxor(hash, pstr[0])
    hash = torch.mmul32(hash, const)
    pstr = pstr + 1
  end
  hash = bit.bxor(hash, 0)
  hash = torch.mmul32(hash, const)
  return hash
end

-- http://www.isthe.com/chongo/tech/comp/fnv/
function torch.CalculateFNV64(char_tensor)
  local pstr = char_tensor:data()
  local const = 1099511628211
  local hash = 14695981039346656037
  while pstr[0] ~= 0 do
    hash = bit.bxor(hash, pstr[0])
    hash = torch.mmul64(hash, const)
    pstr = pstr + 1
  end
  hash = bit.bxor(hash, 0)
  hash = torch.mmul64(hash, const)
  return hash
end

-- Test for the above (32 bit version):
--[[
-- COMPILE AND RUN THIS AT: http://ideone.com/
        #include <stdio.h>
        #include <stdlib.h>

        unsigned int torch.CalculateFNV(const char* str) {
          size_t i;
          const size_t length = strlen(str) + 1;
          unsigned int hash = 2166136261u;
          for (i=0; i<length; ++i) {
            hash ^= *str++;
            hash *= 16777619u;
          }
          return hash;
        }
        int main(void) {
          printf("%s, %u\n", "hello", torch.CalculateFNV("hello"));
          printf("%s, %u\n", "test", torch.CalculateFNV("test"));
          printf("%s, %u\n", "tae34rtou1nbvfopnb234pfiouhwefst",
            torch.CalculateFNV("tae34rtou1nbvfopnb234pfiouhwefst"));
          return 0;
        }

YOU GET:
--> hello, 43209009
--> test, 2854439807
--> tae34rtou1nbvfopnb234pfiouhwefst, 4046593373
--]]
if ffi ~= nil then
  assert(torch.CalculateFNV32(torch.stringToCharTensor('hello')) == 43209009,
         'bad hash')
  assert(torch.CalculateFNV32(torch.stringToCharTensor('test')) == 2854439807,
         'bad hash')
  assert(torch.CalculateFNV32(torch.stringToCharTensor(
      'tae34rtou1nbvfopnb234pfiouhwefst')) == 4046593373, 'bad hash')
end

-- Reads the file into a 2D tensor, where the first dim is the lines
function torch.readFileLinesToCharTensor(filename)
  local nlines = torch.queryNumLinesInFile(filename)
  local max_line_length = torch.queryLongestLineLengthInFile(filename)
  local tensor = torch.CharTensor(nlines, max_line_length + 1)

  -- Diskfile is kinda slow but it doesn't suffer from garbagecollection issues
  -- if the file becomes really big.
  local file = torch.DiskFile(filename, 'r')
  for i = 1, nlines do
    local str = file:readString("*l")
    local ptensor = tensor[{i, {}}]:data()
    ffi.copy(ptensor, str)
  end
  file:close()

--  local i = 1
--  for fn in io.lines(filename) do
--    local ptensor = tensor[{i,{}}]:data()
--    ffi.copy(ptensor, fn)
--    i = i + 1
--  end

  return tensor
end

function torch.queryNumLinesInFile(filename)
  assert(torch.FileExists(filename), 'File does not exist!')
  local handle = io.popen('sed -n $= ' .. filename)
  local result = handle:read("*a")
  handle:close()
  return tonumber(result)
end

function torch.queryLongestLineLengthInFile(filename)
  assert(torch.FileExists(filename), 'File does not exist!')
  local handle = io.popen('wc -L < ' .. filename)
  local result = handle:read("*a")
  handle:close()
  return tonumber(result)
end
-- Test for the above
--[[
x = {'asdfawe','1234gsd$','aw','','546g','6gs'}
x_tensor = stringTableToCharTensor(x)
print(x)
print(x_tensor)
print(torch.charTensorToString(x_tensor[2]))
print(torch.charTensorToStringTable(x_tensor))
--]]

-- NOTE: torch.concatTable is very, very brittle.  Only use it for vectors!
function torch.concatTable(A, B)
  if A == nil and B == nil then
    return nil
  end
  if A == nil then
    return B
  end
  if B == nil then
    return A
  end
  assert(torch.type(A) == 'table' and torch.type(B) == 'table')
  local C = {}
  for i = 1, #A do
    C[i] = A[i]
  end
  for i = 1, #B do
    C[#A + i] = B[i]
  end
  return C
end

function torch.appendTable(table, new_elems)
  assert(torch.type(table) == 'table' and torch.type(new_elems) == 'table')
  for i = 1, #new_elems do
    table[#table + 1] = new_elems[i]
  end
end

function torch.drawText(img, x, y, str, font_size, font_color, italic)
  if font_size == nil then
    font_size = 12
  end
  if italic == nil then
    italic = false
  end
  if font_color == nil then
    font_color = "white"
  end
  local w = qtwidget.newimage(img)
  w:moveto(x, y)
  w:setcolor(font_color)
  w:setfont(qt.QFont{serif = true, italic = italic, size = font_size})
  w:show(str)
  w.port:image():toTensor(img)
end

--[[
-- Test for the above
img = image.scale(image.lena(), 256, 256)
torch.drawText(img, 2, 12, "hello there", 12, "white", false)
torch.drawText(img, 2, 256-2, "hello there", 12, "red", false)
image.display{image=img, zoom=2}
--]]

-- pts is 2 x n (or 3 x n if homogeneous)
-- color is {r, g, b}
-- shape is 'x' or 'o'
function torch.drawUVPts(img, pts, color, shape, size)
  local w = qtwidget.newimage(img)
  w:setlinewidth(1)
  w:setcolor(color[1], color[2], color[3], 1)
  w:sethints('Antialiasing')
  color = color or {1, 0, 0}  -- red
  shape = shape or 'x'
  size = size or 1

  local npts = pts:size(2)
  local dim = math.ceil(size * 0.005 * img:size(3))

  for o = 1, npts do
    local u = pts[{1, o}]
    local v = pts[{2, o}]

    if shape == 'o' then
      -- Draw Circle
      w:arc(u, v, dim, 0, 360)
      w:fill()
    elseif shape == 'x' then
      w:setlinewidth(0.3 * dim)
      -- Draw Cross
      w:moveto(u - dim, v - dim)
      w:lineto(u + dim, v + dim)
      w:stroke()
      w:moveto(u - dim, v + dim)
      w:lineto(u + dim, v - dim)
      w:stroke()
    else
      error('Incorrect shape!')
    end

  end

  w.port:image():toTensor(img)
end

--[[
-- Test for the above
im = image.scale(image.lena(), 512, 512)
pts = torch.FloatTensor(2,2)  -- 2 points
pts[1][1] = 256 -- U1
pts[2][1] = 256 -- V1
pts[1][2] = 128 -- U2
pts[2][2] = 384 -- V2
torch.drawUVPts(im, pts, {1, 0, 0}, 'o', 2)
pts:mul(0.5)
torch.drawUVPts(im, pts, {0, 0, 1}, 'x', 1)
image.display(im)
--]]

-- lines is 2 x 2 x n (or 3 x 2 x n if homogeneous)
-- color is {r, g, b}
function torch.drawUVLines(img, lines, color, size)
  local w = qtwidget.newimage(img)
  w:setlinewidth(1)
  w:setcolor(color[1], color[2], color[3], 1)
  w:sethints('Antialiasing')
  color = color or {1, 0, 0}  -- red
  size = size or 1

  local nlines = lines:size(3)
  local dim = math.ceil(size * 0.005 * img:size(3))

  for o = 1, nlines do
    local u1 = lines[{1, 1, o}]
    local v1 = lines[{2, 1, o}]
    local u2 = lines[{1, 2, o}]
    local v2 = lines[{2, 2, o}]

    w:setlinewidth(0.3 * dim)
    w:moveto(u1, v1)
    w:lineto(u2, v2)
    w:stroke()
  end

  w.port:image():toTensor(img)
end

--[[
-- Test for the above
im = image.scale(image.lena(), 512, 512)
lines = torch.FloatTensor(2,2,2)  -- 2 lines
lines[1][1][1] = 256 -- L1U1
lines[2][1][1] = 256 -- L1V1
lines[1][2][1] = 128 -- L1U2
lines[2][2][1] = 384 -- L1V2
lines[1][1][2] = 1 -- L2U1
lines[2][1][2] = 256 -- L2V1
lines[1][2][2] = 128 -- L2U2
lines[2][2][2] = 384 -- L2V2
torch.drawUVLines(im, lines, {1, 0, 0}, 2)
lines:mul(0.5)
torch.drawUVLines(im, lines, {0, 0, 1}, 1)
image.display(im)
--]]

function torch.setDropoutTrain(model, train)
  local do_nodes = model:findModules('nn.Dropout')
  do_nodes = torch.concatTable(do_nodes, model:findModules('nn.SpatialDropout'))
  for _, dropout in ipairs(do_nodes) do
    dropout.train = train
  end
end

-- This only works for square images
function torch.rotateImageCCW90(im, xdim, ydim)
  assert(xdim <= im:dim() and ydim <= im:dim(), 'Bad xdim and ydim')
  assert(im:size(xdim) == im:size(ydim), 'Image must be square')
  local ret_im = image.flip(im, xdim)
  return ret_im:transpose(xdim, ydim)
end

-- This only works for square images
function torch.rotateImageCW90(im, xdim, ydim)
  assert(xdim <= im:dim() and ydim <= im:dim(), 'Bad xdim and ydim')
  assert(im:size(xdim) == im:size(ydim), 'Image must be square')
  local ret_im = image.flip(im, ydim)
  return ret_im:transpose(xdim, ydim)
end

function torch.rotateImage180(im, xdim, ydim)
  assert(xdim <= im:dim() and ydim <= im:dim(), 'Bad xdim and ydim')
  assert(im:size(xdim) == im:size(ydim), 'Image must be square')
  local ret_im = image.flip(im, ydim)
  return image.flip(ret_im, xdim)
end

-- Test for the above
--[[
im = image.lena()
image.display(im)
image.display{image=torch.rotateImageCCW90(im, 3, 2), legend='CCW90'}
image.display{image=torch.rotateImageCW90(im, 3, 2), legend='CW90'}
image.display{image=torch.rotateImage180(im, 3, 2), legend='180'}
--]]

function torch.deepClone(x, retType)
  local ret
  local xType = torch.type(x)
  if torch.isTensor(x) then
    if xType == retType then
      ret = x:clone()
    else
      ret = x:type(retType)
    end
  elseif xType == 'string' or xType == 'boolean' or xType == 'number' then
    ret = x
  else
    assert(torch.type(x) == 'table')
    ret = {}
    for key, val in pairs(x) do
      ret[key] = torch.deepClone(val, retType) end
  end
  return ret
end

-- Deep-copy all batch data from the CPU to the GPU.
function torch.syncBatchToGPU(batchCPU, batchGPU)
  assert(type(batchCPU) == 'table' and type(batchGPU) == 'table')
  for key, value in pairs(batchCPU) do
    assert(batchGPU[key] ~= nil)
    if type(value) == 'table' then
      syncBatchToGPU(batchGPU[key], value)  -- recurse on sub-tables.
    else
      assert(torch.isTensor(value))
      batchGPU[key]:copy(value)
    end
  end
end

-- Like doing a X(1:stride:end) along a certain dimension in matlab.
function strideTensor(tensor, dim, stride)
  assert(tensor:dim() >= dim)
  local sz = {}
  for i = 1, tensor:dim() do
    if i == dim then
      assert(math.mod(tensor:size(i), stride) == 0,
             'dim not a multiple of stride')
      sz[#sz + 1] = tensor:size(i) / stride
      sz[#sz + 1] = stride
    else
      sz[#sz + 1] = tensor:size(i)
    end
  end
  tensor = tensor:view(unpack(sz))
  return tensor:select(dim + 1, 1):contiguous()
end

function torch.IsInteger(number)
  assert(torch.type(number) == 'number')
  return number == math.floor(number)
end

function torch.HumanReadableNumber(number)
  assert(torch.type(number) == 'number')
  local printFormatStr
  local function formatString(number)
    if torch.IsInteger(number) then
      return '%d'
    else
      return '%.2f'
    end
  end

  local postFix
  if number >= 1e15 then
    number = number * 1e-15
    postFix = 'P'
  elseif number >= 1e12 then
    number = number * 1e-12
    postFix = 'T'
  elseif number >= 1e9 then
    number = number * 1e-9
    postFix = 'G'
  elseif number >= 1e6 then
    number = number * 1e-6
    postFix = 'M' 
  elseif number >= 1e3 then
    number = number * 1e-3
    postFix = 'k'
  elseif number >= 0 then
    postFix = ''
  elseif number >= 1e-3 then
    number = number * 1e3
    postFix = 'm'
  elseif number >= 1e-6 then
    number = number * 1e6
    postFix = 'u'
  elseif number >= 1e-9 then
    number = number * 1e9
    postFix = 'n'
  elseif number >= 1e-12 then
    number = number * 1e12
    postFix = 'p'
  else
    number = number * 1e15
    postFix = 'f'
  end

  if torch.IsInteger(number) then
    return string.format('%d' .. postFix, number)
  else
    return string.format('%.2f' .. postFix, number)
  end
end
