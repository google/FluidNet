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

local tfluids = require('tfluids')

function tfluids.writeOutObstacles(filename, flags)
  assert(#flags:size() == 3)
  local xdim, ydim, zdim = flags:size(1), flags:size(2), flags:size(3)
  -- Single cube data.  Template that will be modified for each cell.
  -- Vertices.
  local v0 = {0.0, 0.0, 0.0}
  local v1 = {1.0, 0.0, 0.0}
  local v2 = {0.0, 1.0, 0.0}
  local v3 = {1.0, 1.0, 0.0}
  local v4 = {0.0, 0.0, 1.0}
  local v5 = {1.0, 0.0, 1.0}
  local v6 = {0.0, 1.0, 1.0}
  local v7 = {1.0, 1.0, 1.0}

  --  Normals.
  local n0 = {0.0, 0.0, -1.0}  -- Face 0 is front. Neg Z direction.
  local n1 = {0.0, 0.0, 1.0}  -- Face 1 is back. Pos Z direction.
  local n2 = {-1.0, 0.0, 0.0}  -- Face 2 is left. Neg X direction.
  local n3 = {1.0, 0.0, 0.0}  -- Face 3 is right. Pos X direction.
  local n4 = {0.0, 1.0, 0.0}  -- Face 4 is top. Pos Y direction.
  local n5 = {0.0, -1.0, 0.0}  -- Face 5 is bottom. Neg Y direction.

  --  Faces. Quads Clockwise ordering!
  local f0 = {0, 2, 3, 1}  -- Face 0 is front.
  local f1 = {4, 5, 7, 6}  -- Face 1 is back.
  local f2 = {4, 6, 2, 0}  -- Face 2 is left.
  local f3 = {1, 3, 7, 5}  -- Face 3 is right.
  local f4 = {2, 6, 7, 3}  -- Face 4 is top.
  local f5 = {0, 1, 5, 4}  -- Face 5 is bottom.

  local cube_points = {v0, v1, v2, v3, v4, v5, v6, v7}
  local cube_normals = {n0, n1, n2, n3, n4, n5}
  local cube_faces = {f0, f1, f2, f3, f4, f5}

  --  Mesh data.
  local points = {}
  local normals = {}
  local faces = {}

  local xscale = 1.0 / xdim
  local yscale = 1.0 / ydim
  local zscale = 1.0 / zdim

  local fluid = tfluids.CellType.TypeFluid

  for z = 1, zdim do
    local zpos = (z - 1) * zscale
    for y = 1, ydim do
      local ypos = (y - 1) * yscale
      for x = 1, zdim do
        local xpos = (x - 1) * xscale
        
        if flags[{x, y, z}] ~= fluid then
          assert(flags[{x, y, z}] == tfluids.CellType.TypeObstacle,
                 'writeOutObstacles can only export fluid + blocker domains.')
        
          --  If we are surounded by obstacles don't write out anything
          local skip = false
          if (z > 1 and z < zdim and x > 1 and x < xdim and y > 1
              and y < ydim) then
              skip = true
              if flags[{x - 1, y, z}] == fluid then skip = false end
              if flags[{x + 1, y, z}] == fluid then skip = false end
              if flags[{x, y - 1, z}] == fluid then skip = false end
              if flags[{x, y + 1, z}] == fluid then skip = false end
              if flags[{x, y, z + 1}] == fluid then skip = false end
              if flags[{x, y, z - 1}] == fluid then skip = false end
          end
          if skip == false then
            --  Remember how many vertices were in the mesh so the offset for 
            --  faces is correct.
            local numverts = #points

            for i = 1, #cube_points do
              local point = {}
              table.insert(point, cube_points[i][1] * xscale + xpos)
              table.insert(point, cube_points[i][2] * yscale + ypos)
              table.insert(point, cube_points[i][3] * zscale + zpos)
              table.insert(points, point)
            end

            for j = 1, #cube_normals do
              table.insert(normals, cube_normals[j])
            end

            for k = 1, #cube_faces do
              local face = {}
              table.insert(face, cube_faces[k][1] + numverts)
              table.insert(face, cube_faces[k][2] + numverts)
              table.insert(face, cube_faces[k][3] + numverts)
              table.insert(face, cube_faces[k][4] + numverts)
              table.insert(faces, face)
            end
          end

        end
      end
    end
  end

  local file = torch.DiskFile(filename, 'w')
  file:writeString("# lots of cubes\n")
  file:writeString("newmtl white_material\n")
  file:writeString("Ka 1.000 1.000 1.000\n")
  file:writeString("Kd 1.000 1.000 1.000\n")
  file:writeString("Ks 0.000 0.000 0.000\n")
  file:writeString("d 1.0\n")
  file:writeString("illum 2\n")
  file:writeString("\n")
  file:writeString("usemtl white_material\n")
  for i = 1, #points do
    local s = string.format("v %.9e %.9e %.9e\n", points[i][1],
        points[i][2], points[i][3])
    file:writeString(s)
  end
  file:writeString("\n")
  for i = 1, #normals do
    local s = string.format("vn %.9e %.9e %.9e\n", normals[i][1],
        normals[i][2], normals[i][3])
    file:writeString(s)
  end
  file:writeString("\n")
  for i = 1, #faces do
    local s = string.format("f %d//%d %d//%d %d//%d  %d//%d\n",
        faces[i][1] + 1, i,  faces[i][2] + 1, i, faces[i][3] + 1, i,
        faces[i][4] + 1, i)
    file:writeString(s)
  end
  file:close()
end
