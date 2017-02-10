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
const real tfluids_(hit_margin) = static_cast<real>(1e-5);
const real tfluids_(epsilon) = static_cast<real>(1e-12);

// Get the integer index of the current voxel.
// Manta defines 0.5 as the center of the first cell, you can see this in
// manta/source/grid.h Grid::getInterpolated() and the lower level call in
// manta/source/util/interpol.h interpol(), where the input position has a
// pos - 0.5 applied to it (our interpol function does this as well).
static inline void GetPixelCenter(const tfluids_(vec3)& pos, int32_t* ix,
                                  int32_t* iy, int32_t* iz) {
  // Note: you could either calculate (int)round(pos.x - 0.5), or you can
  // just round down without taking off the 0.5 value.

  *ix = static_cast<int32_t>(pos.x);
  *iy = static_cast<int32_t>(pos.y);
  *iz = static_cast<int32_t>(pos.z);
}

// Note: IsOutOfDomainReal considers AGAINST the domain to be out of domain.
// It also considers the space from the left of the first cell to the center
// (even though we don't have samples there) and the space from the right of the
// last cell to the border.
static inline bool IsOutOfDomainReal(
    const tfluids_(vec3)& pos, const tfluids_(FlagGrid)& flags) {
  return (pos.x <= (real)0 ||  // LHS of cell.
          pos.x >= (real)flags.xsize() ||  // RHS of cell.
          pos.y <= (real)0 ||
          pos.y >= (real)flags.ysize() ||
          pos.z <= (real)0 ||
          pos.z >= (real)flags.zsize());
}

static inline bool IsBlockedCell(const tfluids_(FlagGrid)& flags,
                                 int32_t i, int32_t j, int32_t k, int32_t b) {
  // Returns true if the cell is blocked.
  // Shouldn't be called on point outside the domain.
  if (flags.isOutOfDomain(i, j, k, b)) {
    // Hard assert here.
    THError("ERROR: IsBlockedCell called on out of domain coords.");
  }
  return !flags.isFluid(i, j, k, b);
}

static inline void tfluids_(ClampToDomain)(
    const tfluids_(FlagGrid)& flags, int32_t* ix, int32_t* iy, int32_t* iz) {
  *ix = std::max<int32_t>(std::min<int32_t>(*ix, flags.xsize() - 1), 0);
  *iy = std::max<int32_t>(std::min<int32_t>(*iy, flags.ysize() - 1), 0);
  *iz = std::max<int32_t>(std::min<int32_t>(*iz, flags.zsize() - 1), 0);
}

static inline void tfluids_(ClampToDomainReal)(
    tfluids_(vec3)& pos, const tfluids_(FlagGrid)& flags) {
  // Clamp to a position epsilon inside the simulation domain. 
  pos.x = std::min<real>(std::max<real>(pos.x, tfluids_(hit_margin)), 
                         (real)flags.xsize() - tfluids_(hit_margin));
  pos.y = std::min<real>(std::max<real>(pos.y, tfluids_(hit_margin)),
                         (real)flags.ysize() - tfluids_(hit_margin));
  pos.z = std::min<real>(std::max<real>(pos.z, tfluids_(hit_margin)),
                         (real)flags.zsize() - tfluids_(hit_margin));
}

// This version takes in the float position, calculates the current voxel index
// and performs the integer lookup on that.
static inline bool IsBlockedCellReal(const tfluids_(FlagGrid)& flags,
                                     const tfluids_(vec3)& pos, int32_t b) {
  int32_t ix, iy, iz;
  GetPixelCenter(pos, &ix, &iy, &iz);
  return IsBlockedCell(flags, ix, iy, iz, b); 
}

#include "quadrants.h"

// I HATE doing this, but I used the code from here:
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
    if (quadrant[i] != MIDDLE && dir[i] != (real)(0.0)) {
      maxT[i] = (candidate_plane[i] - origin[i]) / dir[i];
    } else {
      maxT[i] = (real)(-1.0);
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
  if (maxT[whichPlane] < (real)(0.0)) {
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
bool calcRayBoxIntersection(const tfluids_(vec3)& pos,
                            const tfluids_(vec3)& dt,
                            const tfluids_(vec3)& ctr, 
                            const real hit_margin, tfluids_(vec3)* ipos) {
  if (hit_margin < 0) {
    THError("Error: hit_margin < 0");
  }
  real box_min[3];
  box_min[0] = ctr.x - (real)(0.5) - hit_margin;
  box_min[1] = ctr.y - (real)(0.5) - hit_margin;
  box_min[2] = ctr.z - (real)(0.5) - hit_margin;
  real box_max[3];
  box_max[0] = ctr.x + (real)(0.5) + hit_margin;
  box_max[1] = ctr.y + (real)(0.5) + hit_margin;
  box_max[2] = ctr.z + (real)(0.5) + hit_margin;

  bool hit = HitBoundingBox(box_min, box_max,  // box
                            &pos.x, &dt.x,  // ray
                            &ipos->x);
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
                               const tfluids_(FlagGrid)& flags,
                               const real hit_margin,
                               tfluids_(vec3)* ipos) {
  if (hit_margin <= 0) {
    THError("Error: calcRayBorderIntersection hit_margin < 0.");
  }

  // The source location should be INSIDE the boundary.
  if (IsOutOfDomainReal(pos, flags)) {
    THError("Error: source location is already outside the domain!");
  }
  // The target location should be OUTSIDE the boundary.
  if (!IsOutOfDomainReal(next_pos, flags)) {
    THError("Error: target location is already outside the domain!");
  }

  // Calculate the minimum step length to exit each face and then step that
  // far. The line equation is:
  //   P = gamma * (next_pos - pos) + pos.
  // So calculate gamma required to make P < + margin for each dim
  // independently.
  //   P_i = m --> m - pos_i = gamma * (next_pos_i - pos_i)
  //   --> gamma_i = (m - pos_i) / (next_pos_i - pos_i)
  real min_step = std::numeric_limits<real>::max();
  if (next_pos.x <= hit_margin) {  // left face.
    const real dx = next_pos.x - pos.x;
    if (std::abs(dx) >= tfluids_(epsilon)) {
      const real xstep = (hit_margin - pos.x) / dx;
      min_step = std::min<real>(min_step, xstep);
    }
  }
  if (next_pos.y <= hit_margin) {
    const real dy = next_pos.y - pos.y;
    if (std::abs(dy) >= tfluids_(epsilon)) {
      const real ystep = (hit_margin - pos.y) / dy;
      min_step = std::min<real>(min_step, ystep);
    }
  }
  if (next_pos.z <= hit_margin) {
    const real dz = next_pos.z - pos.z;
    if (std::abs(dz) >= tfluids_(epsilon)) {
      const real zstep = (hit_margin - pos.z) / dz;
      min_step = std::min<real>(min_step, zstep);
    }
  }
  // Also calculate the min step to exit a positive face.
  //   P_i = dim - m --> dim - m - pos_i = gamma * (next_pos_i - pos_i)
  //   --> gamma = (dim - m - pos_i) / (next_pos_i - pos_i)
  if (next_pos.x >= ((real)flags.xsize() - hit_margin)) {  // right face.
    const real dx = next_pos.x - pos.x;
    if (std::abs(dx) >= tfluids_(epsilon)) {
      const real xstep = ((real)flags.xsize() - hit_margin - pos.x) / dx;
      min_step = std::min<real>(min_step, xstep);
    }
  }
  if (next_pos.y >= ((real)flags.ysize() - hit_margin)) {
    const real dy = next_pos.y - pos.y;
    if (std::abs(dy) >= tfluids_(epsilon)) {
      const real ystep = ((real)flags.ysize() - hit_margin - pos.y) / dy;
      min_step = std::min<real>(min_step, ystep);
    }
  }
  if (next_pos.z >= ((real)flags.zsize() - hit_margin)) {
    const real dz = next_pos.z - pos.z;
    if (std::abs(dz) >= tfluids_(epsilon)) {
      const real zstep = ((real)flags.zsize() - hit_margin - pos.z) / dz;
      min_step = std::min<real>(min_step, zstep);
    }
  }
  if (min_step < 0 || min_step >= std::numeric_limits<real>::max()) {
    return false;
  }

  // Take the minimum step.
  ipos->x = min_step * (next_pos.x - pos.x) + pos.x;
  ipos->y = min_step * (next_pos.y - pos.y) + pos.y;
  ipos->z = min_step * (next_pos.z - pos.z) + pos.z;

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
// For real grids values are stored at i+0.5, j+0.5, k+0.5. i.e. the center of
// the first cell is (0.5, 0.5, 0.5) so the corner is (0, 0, 0). Likewise the
// center of the last cell is (xsize - 1 + 0.5, ...) so the corner is
// (xsize, ysize, zsize).
//
// For MAC grids values are stored at i, j+0.5, k+0.5 for the x component.
// So the MAC component for the (i, j, k) index is on the left, bottom and back
// faces of the cell respectively (i.e. the negative edge).
//
// So, if you want to START a line trace at the index (i, j, k) you should add
// 0.5 to each component before calling this function as (i, j, k) converted to
// real will actually be the (left, bottom, back) side of that cell.
bool calcLineTrace(const tfluids_(vec3)& pos, const tfluids_(vec3)& delta,
                   const tfluids_(FlagGrid)& flags, const int32_t ibatch,
                   tfluids_(vec3)* new_pos, const bool do_line_trace) {
  // We can choose to not do a line trace at all.
  if (!do_line_trace) {
    (*new_pos) = pos + delta;
    return false;
  }

  // If we're ALREADY in a obstacle segment (or outside the domain) then a lot
  // of logic below will fail. This function should only be called on fluid
  // cells!
  if (IsOutOfDomainReal(pos, flags)) {
    THError("Error: CalcLineTrace was called on a out of domain cell!");
  }
  if (IsBlockedCellReal(flags, pos, ibatch)) {
    THError("Error: CalcLineTrace was called on a non-fluid cell!");
  }

  (*new_pos) = pos;

  const real length = delta.norm();
  if (length <= tfluids_(epsilon)) {
    // We're not being asked to step anywhere. Return false and copy the pos.
    // (copy already done above).
    return false;
  }
  // Figure out the step size in x, y and z for our marching.
  tfluids_(vec3) dt = delta / length;

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
  real cur_length = (real)0;
  tfluids_(vec3) next_pos;  // Tentative step location.
  while (cur_length < (length - tfluids_(hit_margin))) {
    // We haven't stepped far enough. So take a step.
    real cur_step = std::min<real>(length - cur_length, (real)(1));
    next_pos = (*new_pos) + (dt * cur_step);

    // Check to see if we went too far.
    // TODO(tompson): This is not correct, we might skip over small
    // pieces of geometry if the ray brushes against the corner of a
    // occupied voxel, but doesn't land in it. Fix this (it's very rare though).
  
    // There are two possible cases. We've either stepped out of the domain
    // or entered a blocked cell.
    if (IsOutOfDomainReal(next_pos, flags)) {
      // Case 1. 'next_pos' exits the grid.
      tfluids_(vec3) ipos;
      const bool hit = calcRayBorderIntersection(
          *new_pos, next_pos, flags, tfluids_(hit_margin), &ipos);
      if (!hit) {
        // This is an EXTREMELY rare case. It happens because either the ray is
        // almost parallel to the domain boundary, OR floating point round-off
        // causes the intersection test to fail.
        
        // In this case, fall back to simply clamping next_pos inside the domain
        // boundary. It's not ideal, but better than a hard failure (the reason
        // why it's wrong is that clamping will bring the point off the ray).
        ipos = next_pos;
        tfluids_(ClampToDomainReal)(ipos, flags);
      }

      // Do some sanity checks. I'd rather be slow and correct...
      // The logic above should always put ipos back hit_margin inside the
      // simulation domain.
      if (IsOutOfDomainReal(ipos, flags)) {
        THError("Error: case 1 exited bounds!");
      }

      if (!IsBlockedCellReal(flags, ipos, ibatch)) {
        // OK to return here (i.e. we're up against the border and not
        // in a blocked cell).
        (*new_pos) = ipos;
        return true;
      } else {
        // Otherwise, we hit the border boundary, but we entered a blocked cell.
        // Continue on to case 2.
        next_pos = ipos;
      }
    }
    if (IsBlockedCellReal(flags, next_pos, ibatch)) {
      // Case 2. next_pos enters a blocked cell.
      if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
        // If the source of the ray starts in a blocked cell, we'll never exit
        // the while loop below, also our loop invariant is that new_pos is
        // NEVER allowed to enter a geometry cell. So failing this test means
        // our logic is broken.
        THError("Error: Ray source is already in a blocked cell!");
      }
      const uint32_t max_count = 4;  // TODO(tompson): high enough?
      // Note: we need to spin here because while we backoff a blocked cell that
      // is a unit step away, there might be ANOTHER blocked cell along the ray
      // which is less than a unit step away.
      for (uint32_t count = 0; count <= max_count; count++) {
        if (!IsBlockedCellReal(flags, next_pos, ibatch)) {
          break;
        }
        if (count == max_count) {
          THError("Error: Cannot find non-geometry point (infinite loop)!");
        }

        // Calculate the center of the blocker cell.
        tfluids_(vec3) next_pos_ctr;
        int32_t ix, iy, iz;
        GetPixelCenter(next_pos, &ix, &iy, &iz);
        next_pos_ctr.x = (real)(ix) + (real)0.5;
        next_pos_ctr.y = (real)(iy) + (real)0.5;
        next_pos_ctr.z = (real)(iz) + (real)0.5;

        if (!IsBlockedCellReal(flags, next_pos_ctr, ibatch)) {
          // Sanity check. This is redundant because IsBlockedCellReal USES
          // GetPixelCenter to sample the FlagGrid. But keep this here
          // just in case the implementation changes.
          THError("Error: Center of blocker cell is not a blocker!");
        }
        if (IsOutOfDomainReal(next_pos_ctr, flags)) {
          THError("Error: Center of blocker cell is out of the domain!");
        }
        tfluids_(vec3) ipos;
        const bool hit = calcRayBoxIntersection(*new_pos, dt, next_pos_ctr,
                                                tfluids_(hit_margin), &ipos);
        if (!hit) {
          // This can happen in very rare cases if the ray box
          // intersection test fails because of floating point round off.
          // It can also happen if the simulation becomes unstable (maybe with a
          // poorly trained model) and the velocity values are extremely high.
          
          // In this case, fall back to simply returning new_pos (for which the
          // loop invariant guarantees is a valid point).
          return true;
        }
 
        next_pos = ipos;

        // There's a nasty corner case here. It's when the cell we were trying
        // to step to WAS a blocker, but the ray passed through a blocker to get
        // there (i.e. our step size didn't catch the first blocker). If this is
        // the case we need to do another intersection test, but this time with
        // the ray point destination that is the closer cell.
        // --> There's nothing to do. The outer while loop will try another
        // intersection for us.
      }
      
      // At this point next_pos is guaranteed to be within the domain and
      // not within a solid cell.
      (*new_pos) = next_pos;

      // Do some sanity checks.
      if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
        THError("Error: case 2 entered geometry!");
      }
      if (IsOutOfDomainReal(*new_pos, flags)) {
        THError("Error: case 2 exited bounds!");
      }
      return true;
    }

    // Otherwise, update the position to the current step location.
    (*new_pos) = next_pos;

    // Do some sanity checks and check the loop invariant.
    if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
      THError("Error: correctness assertion broken. Loop entered geometry!");
    }
    if (IsOutOfDomainReal(*new_pos, flags)) {
      THError("Error: correctness assertion broken. Loop exited bounds!");
    }

    cur_length += cur_step;
  }

  // Finally, yet another set of checks, just in case.
  if (IsOutOfDomainReal(*new_pos, flags)) {
    THError("Error: CalcLineTrace returned an out of domain cell!");
  }
  if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
    THError("Error: CalcLineTrace returned a blocked cell!");
  }

  return false;
}

