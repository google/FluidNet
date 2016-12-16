// Copyright 2016 Google Inc, NYU.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <math.h>  // For floor.
#include <limits>

// We never want positions to go exactly to the border or exactly to the edge
// of an occupied piece of geometry. Therefore all rays will be truncated by
// a very small amount (hit_margin).
const real tfluids_(hit_margin) = 1e-5;

static inline bool tfluids_(IsOutOfDomain)(int32_t i, int32_t j, int32_t k,
                                           const Int3& dim) {
  return i < 0 || i >= dim.x || j < 0 || j >= dim.y || k < 0 || k >= dim.z;
}

// Get the integer index of the current voxel.
// Excerpt of comment from tfluids.cc:
// ... the (-0.5, 0, 0) position is the LEFT face of the first cell. Likewise
// (xdim - 0.5, ydim - 0.5, zdim - 0.5) is the upper bound of the grid (right
// at the corner). ...
static inline void GetPixelCenter(const tfluids_(vec3)& pos, int32_t* ix,
                                  int32_t* iy, int32_t* iz) {
  *ix = static_cast<int32_t>(pos.x + static_cast<real>(0.5));
  *iy = static_cast<int32_t>(pos.y + static_cast<real>(0.5));
  *iz = static_cast<int32_t>(pos.z + static_cast<real>(0.5));
}

static inline bool IsOutOfDomainReal(const tfluids_(vec3)& pos,
                                     const Int3& dim) {
  return (pos.x <= static_cast<real>(-0.5) ||  // LHS of grid cell.
          // RHS of grid cell.
          pos.x >= (static_cast<real>(dim.x) - static_cast<real>(0.5)) ||
          pos.y <= static_cast<real>(-0.5) ||
          pos.y >= (static_cast<real>(dim.y) - static_cast<real>(0.5)) ||
          pos.z <= static_cast<real>(-0.5) ||
          pos.z >= (static_cast<real>(dim.z) - static_cast<real>(0.5)));
}

static inline bool IsBlockedCell(const real* obs, int32_t i, int32_t j,
                                 int32_t k, const Int3& dim) {
  // Returns true if the cell is blocked.
  // Shouldn't be called on point outside the domain.
  if (tfluids_(IsOutOfDomain)(i, j, k, dim)) {
    // Hard assert here.
    printf("ERROR: IsBlockedCell called on out of domain coords.\n");
    exit(-1);
  }
  return obs[i + j * dim.x + k * dim.x * dim.y] == static_cast<real>(1);
}

static inline void tfluids_(ClampToDomain)(const Int3& dim, int32_t* ix,
                                           int32_t* iy, int32_t* iz) {
  *ix = std::max<int32_t>(std::min<int32_t>(*ix, dim.x - 1), 0);
  *iy = std::max<int32_t>(std::min<int32_t>(*iy, dim.y - 1), 0);
  *iz = std::max<int32_t>(std::min<int32_t>(*iz, dim.z - 1), 0);
}

static inline void tfluids_(ClampToDomainReal)(tfluids_(vec3)& pos,
                                               const Int3& dim) {
  const real half = static_cast<real>(0.5);
  pos.x = std::min<real>(
      std::max<real>(pos.x, -half + tfluids_(hit_margin)), 
      static_cast<real>(dim.x) - half - tfluids_(hit_margin));
  pos.y = std::min<real>(
      std::max<real>(pos.y, -half + tfluids_(hit_margin)),
      static_cast<real>(dim.y) - half - tfluids_(hit_margin));
  pos.z = std::min<real>(
      std::max<real>(pos.z, -half + tfluids_(hit_margin)),
      static_cast<real>(dim.z) - half - tfluids_(hit_margin));
}

// This version takes in the float position, calculates the current voxel index
// and performs the integer lookup on that.
static inline bool IsBlockedCellReal(const real* obs, const tfluids_(vec3)& pos,
                                     const Int3& dim) {
  int32_t ix, iy, iz;
  GetPixelCenter(pos, &ix, &iy, &iz);
  return IsBlockedCell(obs, ix, iy, iz, dim); 
}

#include "quadrants.h"

// I HATE doing this, but I copied this code from here:
// https://github.com/erich666/GraphicsGems/blob/master/gems/RayBox.c
// And modified it (there were actually a few numerical precision bugs).
// I tested the hell out of it, so it seems to work.
//
// @param hit_margin - value >= 0 describing margin added to hit to
// prevent interpenetration.
bool HitBoundingBox(const real* minB, const real* maxB,  // box
                    const real* origin, const real* dir,  // ray
                    real* coord) {  // hit point.
  char inside = true;
  Quadrants quadrant[3];
  register int i;
  int whichPlane;
  real maxT[3];
  real candidate_plane[3];

  // Find candidate planes; this loop can be avoided if rays cast all from the
  // eye (assume perpsective view).
  for (i = 0; i < 3; i++) {
    if (origin[i] < minB[i]) {
      quadrant[i] = LEFT;
      candidate_plane[i] = minB[i];
      inside = false;
    } else if (origin[i] > maxB[i]) {
      quadrant[i] = RIGHT;
      candidate_plane[i] = maxB[i];
      inside = false;
    } else {
      quadrant[i] = MIDDLE;
    }
  }

  // Ray origin inside bounding box.
  if (inside) {
    for (i = 0; i < 3; i++) {
      coord[i] = origin[i];
    }
    return true;
  }

  // Calculate T distances to candidate planes.
  for (i = 0; i < 3; i++) {
    if (quadrant[i] != MIDDLE && dir[i] != static_cast<real>(0.0)) {
      maxT[i] = (candidate_plane[i] - origin[i]) / dir[i];
    } else {
      maxT[i] = static_cast<real>(-1.0);
    }
  }

  // Get largest of the maxT's for final choice of intersection.
  whichPlane = 0;
  for (i = 1; i < 3; i++) {
    if (maxT[whichPlane] < maxT[i]) {
      whichPlane = i;
    }
  }

  // Check final candidate actually inside box and calculate the coords (if
  // not).
  if (maxT[whichPlane] < static_cast<real>(0.0)) {
    return false;
  }

  const real err_tol = 1e-6;
  for (i = 0; i < 3; i++) {
    if (whichPlane != i) {
      coord[i] = origin[i] + maxT[whichPlane] * dir[i];
      if (coord[i] < (minB[i] - err_tol) || coord[i] > (maxB[i] + err_tol)) {
        return false;
      }
    } else {
      coord[i] = candidate_plane[i];
    }
  }

  return true;
}   

// calcRayBoxIntersection will calculate the intersection point for the ray
// starting at pos, and pointing along dt (which should be unit length).
// The box is size 1 and is centered at ctr.
//
// This is really just a wrapper around the function above.
bool calcRayBoxIntersection(const tfluids_(vec3)& pos,
                            const tfluids_(vec3)& dt,
                            const tfluids_(vec3)& ctr, 
                            const real hit_margin, tfluids_(vec3)& ipos) {
  if (hit_margin < 0) {
    printf("Error: hit_margin < 0\n");
    exit(-1);
  }
  real box_min[3];
  box_min[0] = ctr.x - static_cast<real>(0.5) - hit_margin;
  box_min[1] = ctr.y - static_cast<real>(0.5) - hit_margin;
  box_min[2] = ctr.z - static_cast<real>(0.5) - hit_margin;
  real box_max[3];
  box_max[0] = ctr.x + static_cast<real>(0.5) + hit_margin;
  box_max[1] = ctr.y + static_cast<real>(0.5) + hit_margin;
  box_max[2] = ctr.z + static_cast<real>(0.5) + hit_margin;

  bool hit = HitBoundingBox(box_min, box_max,  // box
                            &pos.x, &dt.x,  // ray
                            &ipos.x);
  return hit;
}

// calcRayBorderIntersection will calculate the intersection point for the ray
// starting at pos and pointing to next_pos.
//
// IMPORTANT: This function ASSUMES that the ray actually intersects. Nasty
// things will happen if it does not.
// EDIT(tompson, 09/25/16): This is so important that we'll actually double
// check the input coords anyway.
bool calcRayBorderIntersection(const tfluids_(vec3)& pos,
                               const tfluids_(vec3)& next_pos,
                               const Int3& dims, const real hit_margin,
                               tfluids_(vec3)& ipos) {
  if (hit_margin < 0) {
    printf("Error: calcRayBorderIntersection hit_margin < 0.\n");
    exit(-1);
  }

  // The source location should be INSIDE the boundary.
  if (IsOutOfDomainReal(pos, dims)) {
    printf("Error: source location is already outside the domain!\n");
    exit(-1);
  }
  // The target location should be OUTSIDE the boundary.
  if (!IsOutOfDomainReal(next_pos, dims)) {
    printf("Error: target location is already outside the domain!\n");
    exit(-1);
  }

  // Calculate the minimum step length to exit each face and then step that
  // far. The line equation is:
  //   P = gamma * (next_pos - pos) + pos.
  // So calculate gamma required to make P < -0.5 + margin for each dim
  // independently.
  //   P_i = -0.5+m --> -0.5+m - pos_i = gamma * (next_pos_i - pos_i)
  //              --> gamma_i = (-0.5+m - pos_i) / (next_pos_i - pos_i)
  real min_step = std::numeric_limits<real>::max();
  if (next_pos.x <= -0.5) {  // left face of cell.
    const real dx = next_pos.x - pos.x;
    if (dx > static_cast<real>(1e-6) || dx < static_cast<real>(-1e-6)) {
      const real xstep = (-0.5 + hit_margin - pos.x) / dx;
      min_step = std::min<real>(min_step, xstep);
    }
  }
  if (next_pos.y <= -0.5) {
    const real dy = next_pos.y - pos.y;
    if (dy > static_cast<real>(1e-6) || dy < static_cast<real>(-1e-6)) {
      const real ystep = (-0.5 + hit_margin - pos.y) / dy;
      min_step = std::min<real>(min_step, ystep);
    }
  }
  if (next_pos.z <= -0.5) {
    const real dz = next_pos.z - pos.z;
    if (dz > static_cast<real>(1e-6) || dz < static_cast<real>(-1e-6)) {
      const real zstep = (-0.5 + hit_margin - pos.z) / dz;
      min_step = std::min<real>(min_step, zstep);
    }
  }
  // Also calculate the min step to exit a positive face.
  //   P_i = dim - 0.5 - m --> dim - 0.5 - m - pos_i =
  //                             gamma * (next_pos_i - pos_i)
  //                       --> gamma = (dim - 0.5 - m - pos_i) /
  //                                   (next_pos_i - pos_i)
  if (next_pos.x >= static_cast<real>(dims.x) - 0.5) {  // right face of cell.
    const real dx = next_pos.x - pos.x;
    if (dx > static_cast<real>(1e-6) || dx < static_cast<real>(-1e-6)) {
      const real xstep =
        (static_cast<real>(dims.x) - 0.5 - hit_margin - pos.x) / dx;
      min_step = std::min<real>(min_step, xstep);
    }
  }
  if (next_pos.y >= static_cast<real>(dims.y) - 0.5) {
    const real dy = next_pos.y - pos.y;
    if (dy > static_cast<real>(1e-6) || dy < static_cast<real>(-1e-6)) {
      const real ystep =
        (static_cast<real>(dims.y) - 0.5 - hit_margin - pos.y) / dy;
      min_step = std::min<real>(min_step, ystep);
    }
  }
  if (next_pos.z >= static_cast<real>(dims.z) - 0.5) {
    const real dz = next_pos.z - pos.z;
    if (dz > static_cast<real>(1e-6) || dz < static_cast<real>(-1e-6)) {
      const real zstep =
        (static_cast<real>(dims.z) - 0.5 - hit_margin - pos.z) / dz;
      min_step = std::min<real>(min_step, zstep);
    }
  }
  if (min_step < 0 || min_step >= std::numeric_limits<real>::max()) {
    return false;
  }

  // Take the minimum step.
  ipos.x = min_step * (next_pos.x - pos.x) + pos.x;
  ipos.y = min_step * (next_pos.y - pos.y) + pos.y;
  ipos.z = min_step * (next_pos.z - pos.z) + pos.z;

  return true;
}

// The following function performs a line trace along the displacement vector
// and returns either:
// a) The position 'p + delta' if NO geometry is found on the line trace. or
// b) The position at the first geometry blocker along the path.
// The search is exhaustive (i.e. O(n) in the length of the displacement vector)
//
// Note: the returned position is NEVER in geometry or outside the bounds. We
// go to great lengths to ensure this.
//
// TODO(tompsion): This is probably not efficient at all.
// It also has the potential to miss geometry along the path if the width
// of the geometry is less than 1 grid.
//
// NOTE: Pos = (0, 0, 0) is the CENTER of the first grid cell.
//       Pos = (0.5, 0, 0) is the x face between the first 2 cells.
bool calcLineTrace(const tfluids_(vec3)& pos, const tfluids_(vec3)& delta,
                   const Int3& dims, const real* obs, tfluids_(vec3)& new_pos) {
  // If we're ALREADY in a geometry segment (or outside the domain) then a lot
  // of logic below with fail. This function should only be called on fluid
  // cells!
  if (IsOutOfDomainReal(pos, dims) || IsBlockedCellReal(obs, pos, dims)) {
    printf("Error: CalcLineTrace was called on a non-fluid cell!\n");
    exit(-1);
  }

  new_pos.x = pos.x;
  new_pos.y = pos.y;
  new_pos.z = pos.z;

  const real length = tfluids_(length3)(delta);
  if (length <= static_cast<real>(1e-6)) {
    // We're not being asked to step anywhere. Return false and copy the pos.
    // (copy already done above).
    return false;
  }
  // Figure out the step size in x, y and z for our marching.
  tfluids_(vec3) dt;
  dt.x = delta.x / length;  // Recall: we've already check for div zero.
  dt.y = delta.y / length;
  dt.z = delta.z / length;

  // Otherwise, we start the line search, by stepping a unit length along the
  // vector and checking the neighbours.
  //
  // A few words about the implementation (because it's complicated and perhaps
  // needlessly so). We maintain a VERY important loop invariant: new_pos is
  // NEVER allowed to enter solid geometry or go off the domain. next_pos
  // is the next step's tentative location, and we will always try and back
  // it off to the closest non-geometry valid cell before updating new_pos.
  //
  // We will also go to great lengths to ensure this loop invariant is
  // correct (probably at the expense of speed).
  real cur_length = 0;
  tfluids_(vec3) next_pos;  // Tentative step location.
  while (cur_length < (length - tfluids_(hit_margin))) {
    // We haven't stepped far enough. So take a step.
    real cur_step = std::min<real>(length - cur_length, static_cast<real>(1));
    next_pos.x = new_pos.x + cur_step * dt.x;
    next_pos.y = new_pos.y + cur_step * dt.y;
    next_pos.z = new_pos.z + cur_step * dt.z;

    // Check to see if we went too far.
    // TODO(tompson): This is not correct, we might skip over small
    // pieces of geometry if the ray brushes against the corner of a
    // occupied voxel, but doesn't land in it. Fix this (it's very rare though).
  
    // There are two possible cases. We've either stepped out of the domain
    // or entered a blocked cell.
    if (IsOutOfDomainReal(next_pos, dims)) {
      // Case 1. 'next_pos' exits the grid.
      tfluids_(vec3) ipos;
      const bool hit = calcRayBorderIntersection(new_pos, next_pos, dims, 
                                                 tfluids_(hit_margin), ipos);
      if (!hit) {
        // This is an EXTREMELY rare case. It happens once or twice during
        // training. It happens because either the ray is almost parallel
        // to the domain boundary, OR floating point round-off causes the
        // intersection test to fail.
        
        // In this case, fall back to simply clamping next_pos inside the domain
        // boundary. It's not ideal, but better than a hard failure.
        ipos.x = next_pos.x;
        ipos.y = next_pos.y;
        ipos.z = next_pos.z;
        tfluids_(ClampToDomainReal)(ipos, dims);
      }

      // Do some sanity checks. I'd rather be slow and correct...
      // The logic above should aways put ipos back inside the simulation
      // domain.
      if (IsOutOfDomainReal(ipos, dims)) {
        printf("Error: case 1 exited bounds!\n");
        exit(-1);
      }

      if (!IsBlockedCellReal(obs, ipos, dims)) {
        // OK to return here (i.e. we're up against the border and not
        // in a blocked cell).
        new_pos.x = ipos.x;
        new_pos.y = ipos.y;
        new_pos.z = ipos.z;
        return true;
      } else {
        // Otherwise, we hit the border boundary, but we entered a blocked cell.
        // Continue on to case 2.
        next_pos.x = ipos.x;
        next_pos.y = ipos.y;
        next_pos.z = ipos.z;
      }
    }
    if (IsBlockedCellReal(obs, next_pos, dims)) {
      // Case 2. next_pos enters a blocked cell.
      if (IsBlockedCellReal(obs, new_pos, dims)) {
        // If the source of the ray starts in a blocked cell, we'll never exit
        // the while loop below, also our loop invariant is that new_pos is
        // NEVER allowed to enter a geometry cell. So failing this test means
        // our logic is broken.
        printf("Error: Ray source is already in a blocked cell!\n");
        exit(-1);
      }
      uint32_t count = 0;
      const uint32_t max_count = 100;
      while (IsBlockedCellReal(obs, next_pos, dims)) {
        // Calculate the center of the blocker cell.
        tfluids_(vec3) next_pos_ctr;
        int32_t ix, iy, iz;
        GetPixelCenter(next_pos, &ix, &iy, &iz);
        next_pos_ctr.x = static_cast<real>(ix);
        next_pos_ctr.y = static_cast<real>(iy);
        next_pos_ctr.z = static_cast<real>(iz);
        if (!IsBlockedCellReal(obs, next_pos_ctr, dims)) {
          // Sanity check. This is redundant because IsBlockedCellReal USES
          // GetPixelCenter to sample the geometry field. But keep this here
          // just in case the implementation changes.
          printf("Error: Center of blocker cell is not a blocker!\n");
          exit(-1);
        }
        if (IsOutOfDomainReal(next_pos_ctr, dims)) {
          printf("Error: Center of blocker cell is out of the domain!\n");
          exit(-1);
        }
        tfluids_(vec3) ipos;
        const bool hit = calcRayBoxIntersection(new_pos, dt, next_pos_ctr,
                                                tfluids_(hit_margin), ipos);
        // Hard assert if we didn't hit (even on release builds) because we
        // should have hit the aabbox!
        if (!hit) {
          // EDIT: This can happen in very rare cases if the ray box
          // intersection test fails because of floating point round off.
          // It can also happen if the simulation becomes unstable (maybe with a
          // poorly trained model) and the velocity values are extremely high.
          
          // In this case, fall back to simply returning new_pos (for which the
          // loop invariant guarantees is a valid point).
          next_pos.x = new_pos.x;
          next_pos.y = new_pos.y;
          next_pos.z = new_pos.z;
          return true;
        }
  
        next_pos.x = ipos.x;
        next_pos.y = ipos.y;
        next_pos.z = ipos.z;

        // There's a nasty corner case here. It's when the cell we were trying
        // to step to WAS a blocker, but the ray passed through a blocker to get
        // there (i.e. our step size didn't catch the first blocker). If this is
        // the case we need to do another intersection test, but this time with
        // the ray point destination that is the closer cell.
        // --> There's nothing to do. The outer while loop will try another
        // intersection for us.

        // A basic test to make sure we never spin here indefinitely (I've
        // never seen it, but just in case).
        count++;
        if (count >= max_count) {
          printf("Error: Cannot find non-geometry point (infinite loop)!");
          exit(-1);
        }
      }
      
      // At this point next_pos is guaranteed to be within the domain and
      // not within a solid cell. 
      new_pos.x = next_pos.x;
      new_pos.y = next_pos.y;
      new_pos.z = next_pos.z;

      // Do some sanity checks.
      if (IsBlockedCellReal(obs, new_pos, dims)) {
        printf("Error: case 2 entered geometry!\n");
        exit(-1);
      }
      if (IsOutOfDomainReal(new_pos, dims)) {
        printf("Error: case 2 exited bounds!\n");
        exit(-1);
      }
      return true;
    }

    // Otherwise, update the position to the current step location.
    new_pos.x = next_pos.x;
    new_pos.y = next_pos.y;
    new_pos.z = next_pos.z;

    // Do some sanity checks and check the loop invariant.
    if (IsBlockedCellReal(obs, new_pos, dims)) {
      printf("Error: correctness assertion broken. Loop entered geometry!\n");
      exit(-1);
    }
    if (IsOutOfDomainReal(new_pos, dims)) {
      printf("Error: correctness assertion broken. Loop exited bounds!\n");
      exit(-1);
    }

    cur_length += cur_step;
  }

  // Finally, yet another set of checks, just in case.
  if (IsOutOfDomainReal(new_pos, dims)) {
    printf("Error: CalcLineTrace returned an out of domain cell!\n");
    exit(-1);
  }
  if (IsBlockedCellReal(obs, new_pos, dims)) {
    printf("Error: CalcLineTrace returned a blocked cell!\n");
    exit(-1);
  }

  return false;
}

