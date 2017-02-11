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

int findConnectedFluidComponents(
    const tfluids_(FlagGrid)& flags, THIntTensor* components,
    const int32_t ibatch, std::vector<int32_t>* component_sizes) {
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const bool is_3d = flags.is_3d();
  int cur_component = 0;
  std::vector<Int3> stack;
  THIntTensor_fill(components, -1);  // -1 means non-fluid and not yet procesed.

  for (int32_t k = 0; k < zsize; k++) {
    for (int32_t j = 0; j < ysize; j++) {
      for (int32_t i = 0; i < xsize; i++) {
        if (THIntTensor_get3d(components, k, j, i) < 0) {
          if (THIntTensor_get3d(components, k, j, i) != -1) {
            THError("INTERNAL ERROR: visited neighbor wasn't processed!");
          }
          if (flags.isFluid(i, j, k, ibatch)) {
            // We haven't processed the current cell, push it on the stack and
            // process the component.
            stack.push_back(Int3(i, j, k));
            component_sizes->push_back(0);
            while (!stack.empty()) {
              const Int3 p = stack.back();
              stack.pop_back();
              THIntTensor_set3d(components, p.z, p.y, p.x, cur_component);
              (*component_sizes)[cur_component]++;
              // Process the 4 or 6 (for 2D and 3D) neighbors.
              const Int3 neighbors[6] = {
                  Int3(p.x - 1, p.y, p.z),
                  Int3(p.x + 1, p.y, p.z),
                  Int3(p.x, p.y - 1, p.z),
                  Int3(p.x, p.y + 1, p.z),
                  Int3(p.x, p.y, p.z - 1),
                  Int3(p.x, p.y, p.z + 1)
              };
              const int32_t nneighbors = is_3d ? 6 : 4;
              for (int32_t n = 0; n < nneighbors; n++) {
                const Int3& pn = neighbors[n];
                if (pn.x >= 0 && pn.x < xsize &&
                    pn.y >= 0 && pn.y < ysize &&
                    pn.z >= 0 && pn.z < zsize) {
                  if (flags.isFluid(pn.x, pn.y, pn.z, ibatch) &&
                      THIntTensor_get3d(components, pn.z, pn.y, pn.x) == -1) {
                    // Neighbor is fluid and hasn't been visited.
                    // Mark as visited but not yet processed and add it to the
                    // stack for later processing.
                    THIntTensor_set3d(components, pn.z, pn.y, pn.x, -2);
                    stack.push_back(Int3(pn.x, pn.y, pn.z));
                  }
                }
              }
            }  // !stack.empty
            // We've finished processing the current component.
            cur_component++;
          } else {
            // The cell is non-fluid and wont be included in any PCG system.
            // Leave the component index at -1.
          }
        }
      }
    }
  }
  return cur_component;
}

