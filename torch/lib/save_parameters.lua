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

-- tableToString, recursively prints a table to a string.
function torch.tableToString(val, name, skipNewLines, skipName, depth)
  skipNewLines = skipNewLines or false
  skipName = skipName or false
  depth = depth or 0

  local tmp = string.rep("  ", depth)

  if name and (not skipName) then
    tmp = tmp .. name .. " = "
  end

  if type(val) == "table" then
    tmp = tmp .. "{" .. (not skipNewLines and "\n" or "")

    -- Get a list of keys in the table.
    local keys = {}
    for k, v in pairs(val) do
      keys[#keys + 1] = k
    end

    -- Sort the keys alphanumerically. This makes doing diff against output
    -- consistent (otherwise it will be presented by hash-set order, which is
    -- not well defined).
    table.sort(keys)

    for i = 1, #keys do
      local k = keys[i]
      local v = val[k]
      tmp =
          tmp .. torch.tableToString(v, k, skipNewLines, skipName, depth + 1) ..
              "," .. (not skipNewLines and "\n" or "")
    end

    tmp = tmp .. string.rep("  ", depth) .. "}"
  elseif type(val) == "number" then
    tmp = tmp .. tostring(val)
  elseif type(val) == "string" then
    tmp = tmp .. string.format("%q", val)
  elseif type(val) == "boolean" then
    tmp = tmp .. (val and "true" or "false")
  else
    tmp = tmp .. "\"[inserializeable datatype:" .. type(val) .. "]\""
  end

  return tmp
end

local function saveStringToFile(filename, str)
  local f = assert(io.open(filename, "w"))
  f:write(str)
  f:close()
end

function torch.saveParameters(conf, mconf)
  local confStr = torch.tableToString(conf)
  local mconfStr = torch.tableToString(mconf)
  print('==> CONF:\n' .. confStr)
  print('==> MCONF:\n' .. mconfStr)
  saveStringToFile(conf.modelDirname .. '_conf.txt', confStr)
  saveStringToFile(conf.modelDirname .. '_mconf.txt', mconfStr)
end

