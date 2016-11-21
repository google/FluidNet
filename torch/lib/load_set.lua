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

local paths = require('paths')

function torch.loadSet(conf, dataType)
  local data
  local filename = conf.cacheDir .. '/preprocessed_' .. conf.dataset .. '_' ..
                   dataType .. '.bin'
  if paths.filep(filename) then
    print('Loading preprocessed file ' .. filename)
    data = torch.load(filename)
  end
  if data == nil then
    print('Loading ' .. dataType .. ' from disk')
    -- We couldn't load the data from disk.
    data = torch.DataBinary(conf, dataType)
    torch.save(filename, data)
  end
  print('==> Loaded ' .. data:nsamples() .. ' samples')
  return data
end
