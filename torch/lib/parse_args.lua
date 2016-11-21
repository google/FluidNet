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

-- This method is a replacement for the API of OptionsParser (xlua). This
-- framework enables parsing of the command-line options and a very easy API to
-- set defaults (with basic type checking); see usage example below for details.
--
-- NOTE: Nested options are separated by '.' character so this is reserved.
--
-- USAGE LUA:
--[[
options = {
  aNumber = 20,
  importantStuff = {
    importantString = '42',
    importantBool = false
  }
}
options = torch.parseArgs(options)
--]]
-- USAGE COMMAND LINE:
--[[
>> qlua myscript.lua -aNumber 30 -importantStuff.importantString hello
>> qlua myscript.lua -help
--]]

local function checkOptionsTable(optionsTable)
  for key, value in pairs(optionsTable) do
    assert(string.find(key, '%.') == nil, "ERROR: '.' character is reserved")
    if type(value) == 'table' then
      checkOptionsTable(value)
    else
      assert(type(key) == 'string' or
             type(key) == 'boolean' or
             type(key) == 'number',
             'Options must be strings, booleans or numbers')
      if type(key) == 'string' then
        assert(key ~= 'help', 'Help keyword is reserved')
      end
    end
  end
end

local function printFailure(str)
  io.stderr:write(str .. '\n')
  os.exit(1)
end

local function printOptionsTable(optionsTable, prefix)
  for key, value in pairs(optionsTable) do
    if type(value) == 'table' then
      if prefix ~= nil then
        printOptionsTable(value, prefix .. key .. '.')
      else
        printOptionsTable(value, key .. '.')
      end
    else
      if prefix ~= nil then
        print('  -' .. prefix .. key .. ' (' .. tostring(value) .. ')')
      else
        print('  -' .. key .. ' (' .. tostring(value) .. ')')
      end
    end
  end
end

local function isInt(n)
  return n == math.floor(n)
end

function torch.parseArgs(optionsTable)
  checkOptionsTable(optionsTable)

  if arg == nil then
    print("WARNING: No input arguments! Likely you're calling this function" ..
          " from the interpreter.")
    return optionsTable
  end

  -- Parse the command line options.
  -- expand options (e.g. "--input=file" -> "--input", "file").
  local arg = {unpack(arg)}
  for i = #arg, 1, -1 do local v = arg[i]
    local flag, val = v:match('^(%-%-%w+)=(.*)')
    if flag then
      arg[i] = flag
      table.insert(arg, i + 1, val)
    end
  end

  -- When calling from the interp. of DeepMind's torch, there is always a
  -- single argument 'ltorch' which we should ignore.
  if #arg == 1 and arg[1] == '-ltorch' then
    print("WARNING: No input arguments! Likely you're calling this function" ..
          " from the interpreter.")
    return optionsTable
  end

  local i = 1
  for i = 1, #arg, 2 do
    local key = arg[i]
    if key:match('^%-') == nil then
      printFailure("ERROR: option does not have '-' character prefix: " ..
                   arg[i])
    end
    key = key:sub(2)  -- Remove '-' prefix
    if key == 'help' or key == '-help' then
      print('avaliable options (with defaults):')
      printOptionsTable(optionsTable, nil)
      os.exit()
    end
    if #arg < (i + 1) then
      printFailure("ERROR: option " .. key ..
                   " was given no corresponding value")
    end
    local value = arg[i + 1]

    local keys = string.split(key, '.')

    local function lookupKey(curTable, key)
       -- Note, the key can be string '1' or also sometimes numeric as well!
      local numericKey = tonumber(key)
      if numericKey ~= nil and isInt(numericKey) == false then
        numericKey = nil  -- Don't accept fractional keys
      end
      if curTable[key] ~= nil then
        return curTable[key]
      elseif curTable[numericKey] ~= nil then
        return curTable[numericKey]
      else
        return nil
      end
    end

    -- Check that the option's nested table exists.
    local curTable = optionsTable
    for i = 1, #keys - 1 do
      curTable = lookupKey(curTable, keys[i])

      if curTable == nil then
        printFailure("ERROR: user-defined option " .. arg[i] .. "=" .. value ..
                     " is not a valid option")
      end
    end

    -- Check that the option exists.
    if lookupKey(curTable, keys[#keys]) == nil then
      printFailure("ERROR: user-defined option " .. arg[i] .. "=" .. value ..
                   " is not a valid option")
    end

    if curTable[keys[#keys]] == nil then
      -- The option key must be numeric.
      keys[#keys] = tonumber(keys[#keys])
    end

    -- Convert the value to the type in the current optionsTable.
    local dstType = type(curTable[keys[#keys]])
    if dstType == 'table' then
      printFailure("ERROR: user-defined option " .. arg[i] .. "=" .. value ..
                   " is not a valid option. You cannot set whole table " ..
                   "options from the command line. Instead use [n] syntax.")
    elseif dstType == 'number' then
      local numValue = tonumber(value)
      if numValue == nil then
        printFailure("ERROR: user-defined option " .. arg[i] .. "=" .. value ..
                     " must be a number")
      end
      value = numValue
    elseif dstType == 'boolean' then
      if value == 'true' then
        value = true
      elseif value == 'false' then
        value = false
      else
        printFailure("ERROR: user-defined option " .. arg[i] .. "=" .. value ..
                     " must be true/false")
      end
    elseif dstType == 'string' then
      -- key is already a string, but make sure the input is NOT a number.
      if tonumber(value) ~= nil then
        printFailure("ERROR: user-defined option " .. arg[i] .. "=" .. value ..
                     " must be a string")
      end
    end

    curTable[keys[#keys]] = value
  end

  return optionsTable
end
