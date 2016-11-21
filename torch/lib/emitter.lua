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

local emitter = {}  -- package namespace.

-- TODO(tompson,kris): This is not how you define classes in torch. You should
-- use the torch class system.
local vec3 = {}
emitter.vec3 = vec3
vec3.__index = vec3

function vec3.create(x, y, z)
  local v = {}
  setmetatable(v, vec3)
  v.x = x
  v.y = y
  v.z = z
  return v
end

-- Operator overload.
function vec3.__add(a, b)
  local v = vec3.create(
      a.x + b.x,
      a.y + b.y,
      a.z + b.z
      )
  return v
end

function vec3:set(x, y, z)
  self.x = x
  self.y = y
  self.z = z
  return self
end

-- Test operator overloading.
do
  local x = vec3.create(1, 2, 3)
  local y = vec3.create(-1, -2, -3)
  local z = x + y

  assert(z.x == 0 and z.y == 0 and z.z == 0, "vec3 add didn't work")
end

local Vec3Utils = {}
emitter.Vec3Utils = Vec3Utils
Vec3Utils.__index = Vec3Utils

function Vec3Utils.clone(a)
  local v = vec3.create(a.x, a.y, a.z)
  return v
end

function Vec3Utils.dot(a, b)
  return a.x * b.x + a.y * b.y + a.z * b.z
end

function Vec3Utils.lengthSquared( v )
  return Vec3Utils.dot(v, v)
end

function Vec3Utils.length(v)
  local len_sq = Vec3Utils.lengthSquared(v)
  if len_sq > 1e-7 then
    return math.sqrt(len_sq)
  else
    return 0
  end
end

function Vec3Utils.setLength(source, s)
  local len = Vec3Utils.length(source)
  if len > 1e-6 then
    source.x = source.x / len
    source.y = source.y / len
    source.z = source.z / len

    source.x = source.x * s
    source.y = source.y * s
    source.z = source.z * s
  end
  return source
end

function Vec3Utils.rotateZAxis(v, angleInRadians)
  v.x = v.x * math.cos(angleInRadians) - v.y * math.sin(angleInRadians)
  v.y = v.x * math.sin(angleInRadians) + v.y * math.cos(angleInRadians)
  return v
end

function Vec3Utils.rotateXAxis(v, angleInRadians)
  v.y = v.y * math.cos(angleInRadians) - v.z * math.sin(angleInRadians)
  v.z = v.y * math.sin(angleInRadians) + v.z * math.cos(angleInRadians)
  return v
end

function Vec3Utils.rotateYAxis(v, angleInRadians)
  v.z = v.z * math.cos(angleInRadians) - v.x * math.sin(angleInRadians)
  v.x = v.z * math.sin(angleInRadians) + v.x * math.cos(angleInRadians)
  return v
end

local Sphere = {}
emitter.Sphere = Sphere
Sphere.__index = Sphere

function Sphere.create(location, radius)
  local sp = {} --new object
  setmetatable(sp, Sphere) --new object handle lookup
  sp.center = location
  sp.radius = radius
  return sp
end

function Sphere:signedDistance(pos)
  local dx = pos.x - self.center.x
  local dy = pos.y - self.center.y
  local dz = pos.z - self.center.z
  local len_sq = dx * dx + dy * dy + dz * dz
  local len = 0
  if len_sq > 1e-7 then
    len = math.sqrt(len_sq)
  end
  return len - self.radius
end

local function isInside(p)
  local dist = self:signedDistance(p)
  return dist <= 0.0
end

local MathUtils = {}
emitter.MathUtils = MathUtils
MathUtils.__index = MathUtils

-- Static method call is just object.method().
-- using the colon is the equivalent to object.method(object)
-- object:method() == object.method(object)
function MathUtils.sign(x)
  if x < 0.0 then return -1.0 end
  return 1.0
end

function MathUtils.smoothstep(edge0, edge1, x)
  x = MathUtils.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return x * x * (3 - 2 * x)
end

function MathUtils.clamp(x, a, b)
  if x < a then
    return a
  end
  if x > b then
    return b
  end
  return x
end

function MathUtils.sphereForceFalloff(sphere, location)
  local signedDist = sphere:signedDistance(location)
  if signedDist < 0.0 then
    local t = signedDist / -sphere.radius
    return MathUtils.smoothstep(0.0, 1.0, t)
  end
  return 0.0
end

return emitter
