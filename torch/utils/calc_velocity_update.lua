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

local torch = require('torch')

-- Velocity update to mimic CorrectVelocity() in manta (line 57 of
-- source/plugin/pressure.cpp). This function is slow and is JUST for testing.
--
-- @param deltaU is the output (i.e. delta V). It's basically a pressure
-- gradient with some boundary conditions handling.
-- @param matchManta - boolean. If true then include manta's idiosyncrasies
-- (which means the output might not be correct but will match the gradient
-- update from Manta).
function torch.calcVelocityUpdate(deltaU, p, geom, matchManta)
  assert(geom:dim() == 3 and p:dim() == 3)
  local xsize = geom:size(3)
  local ysize = geom:size(2)
  local zsize = geom:size(1)
  local twoDim = deltaU:size(1) == 2
  if twoDim then
    assert(zsize == 1, '2D velocity tensor but zsize ~= 1')
  end

  -- Centered cell grid so negate the component inside the obstacle.
  for z = 1, zsize do
    for y = 1, ysize do
      for x = 1, xsize do
        if geom[{z, y, x}] == 1 then
          -- The current cell is geometry. It should not receive a velocity
          -- update.
          deltaU[{{}, z, y, x}]:fill(0)
        else
          -- The current cell is a fluid cell and needs a velocity update.
          local function calcPartialDeriv(pos, dim, size)
            -- All comments are as if dim is the x dimension. But they hold for
            -- the others as well. (i.e. left/right = top/bottom | front/back).
            local posPos = pos:clone()
            posPos[dim] = posPos[dim] + 1  -- To the right.
            local posNeg = pos:clone()
            posNeg[dim] = posNeg[dim] - 1  -- To the left.

            if (pos[dim] == 1) and matchManta then
              -- First annoying special case that happens on the border because
              -- of our conversion to central velocities and because manta does 
              -- not handle this case properly.
              if (geom[{posPos[3], posPos[2], posPos[1]}] == 1) and
                  (geom[{pos[3], pos[2], pos[1]}] == 0) then
                deltaU[{dim, pos[3], pos[2], pos[1]}] =
                    p[{pos[3], pos[2], pos[1]}] * 0.5
              else
                deltaU[{dim, pos[3], pos[2], pos[1]}] =
                    p[{posPos[3], posPos[2], posPos[1]}] * 0.5
              end
              return
            end

            -- Look at the neighbour to the right (pos) and to the left (neg).
            local geomPos, geomNeg = false, false
            if pos[dim] == 1 then
              geomNeg = true  -- Treat going off the fluid as geometry.
            end
            if pos[dim] == size[dim] then
              geomPos = true  -- Treat going off the fluid as geometry. 
            end
            if pos[dim] > 1 then
              geomNeg = geom[{posNeg[3], posNeg[2], posNeg[1]}] == 1
            end
            if pos[dim] < size[dim] then
              geomPos = geom[{posPos[3], posPos[2], posPos[1]}] == 1
            end

            -- NOTE: The 0.5 below needs some explanation. We are exactly
            -- mimicking CorrectVelocity() from
            -- manta/source/pluging/pressure.cpp. In this function, all
            -- updates are single sided, but they are done to the MAC cell
            -- edges. When we convert to centered velocities, we therefore add
            -- a * 0.5 term because we take the average.
            local singleSidedGain = 1
            if matchManta then
              singleSidedGain = 0.5
            end

            if geomPos and geomNeg then
              -- There are 3 cases:
              -- A) Cell is on the left border and has a right geom neighbor.
              -- B) Cell is on the right border and has a left geom neighbor.
              -- C) Cell has a right AND left geom neighbor.
              -- In any of these cases the velocity should not receive a
              -- pressure gradient (nowhere for the pressure to diffuse.
              deltaU[{dim, pos[3], pos[2], pos[1]}] = 0
            elseif geomPos then
              -- There are 2 cases:
              -- A) Cell is on the right border and there's fluid to the left.
              -- B) Cell is internal but there is geom to the right.
              -- In this case we need to do a single sided diff to the left.
              deltaU[{dim, pos[3], pos[2], pos[1]}] =
                  (p[{pos[3], pos[2], pos[1]}] -
                   p[{posNeg[3], posNeg[2], posNeg[1]}]) * singleSidedGain
            elseif geomNeg then
              -- There are 2 cases:
              -- A) Cell is on the left border and there's fluid to the right.
              -- B) Cell is internal but there is geom to the left.
              -- In this case we need to do a single sided diff to the right.
              deltaU[{dim, pos[3], pos[2], pos[1]}] =
                  (p[{posPos[3], posPos[2], posPos[1]}] -
                   p[{pos[3], pos[2], pos[1]}]) * singleSidedGain
            else
              -- The pixel is internal (not on border) with no geom neighbours.
              -- Do a central diff.
              deltaU[{dim, pos[3], pos[2], pos[1]}] =
                  (p[{posPos[3], posPos[2], posPos[1]}] -
                   p[{posNeg[3], posNeg[2], posNeg[1]}]) * 0.5
            end
          end

          local pos = torch.FloatTensor({x, y, z})
          local size = torch.FloatTensor({xsize, ysize, zsize})
          calcPartialDeriv(pos, 1, size)
          calcPartialDeriv(pos, 2, size)
          if not twoDim then
            calcPartialDeriv(pos, 3, size)
          end
        end
      end
    end
  end

  local invGeom = geom:clone():mul(-1):add(1)
end

