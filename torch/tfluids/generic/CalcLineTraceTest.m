% Copyright 2016 Google Inc, NYU.
% 
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
% 
%     http://www.apache.org/licenses/LICENSE-2.0
% 
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% Just a hacky playground for making sure the line tracing routine is
% correct by visualizing it.
% To run this you must first run "CompileMex.m"

% NOTE: Before running this script you will need to download the 
% VoxelPlotter matlab utility from:
% https://www.mathworks.com/matlabcentral/fileexchange/50802-voxelplotter
% and put the source code in this directory.
function [] = CalcLineTraceTest()
  close all; clc; clearvars;
  CalcLineTraceCompileMex();
  
  % First test a 2 voxel grid.
  dims = [3, 4, 5];
  obs = zeros(dims(1), dims(2), dims(3));
  filled_cell = [1, 2, 3];
  obs(filled_cell(1) + 1, filled_cell(2) + 1, filled_cell(3) + 1) = 1;
  filled_cell = filled_cell + 0.5;
  figure;
  set(gcf, 'Position', [100 100 1000 1000]);
  h = VoxelPlotter(obs, 1);
  view(45, 45); hold on; grid on;
  axis vis3d;
  axis([0, dims(1), 0, dims(2), 0, dims(3)]);
  h.Vertices = h.Vertices - 0.5;  % Need cells centered at 0.5.
  
  % Make a starting point at EVERY grid center except the occupied grid.
  pos_vals = zeros(dims(1) * dims(2) * dims(3) - 1, 3);
  iout = 1;
  for z = 1:dims(3)
    for y = 1:dims(2)
      for x = 1:dims(1)
        pos = ([x, y, z] - 1) + 0.5;  % 0-indexed.
        if norm(filled_cell - pos) > 1e-5
          pos_vals(iout, :) = pos;  % 0.5 is the center of the cell
          iout = iout + 1;
        end
      end
    end
  end
  for i = 1:size(pos_vals, 1)
    pos = pos_vals(i, :);
    delta = 0.9 * (filled_cell - pos) - [0.001, 0, 0];
    [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
    assert(collide == 1);
  end
  
  % Now test a more complicated setup with a larger voxel.
  depth = 28;
  width = 26;
  height = 33;
  dims = [width, height, depth];
  
  % Make some dummy geometry --> Just a circle.
  circle_center = [width/2, height/2, depth/2];  % (x, y, z)
  circle_radius = min([width, depth, height]) / 4 + 0.5;
  obs = zeros(width, height, depth);  
  [u, v, z] = ndgrid(1:width, 1:height, 1:depth);
  u = u - circle_center(1);
  v = v - circle_center(2);
  z = z - circle_center(3);
  obs = (u .* u + v .* v + z .* z) <= circle_radius * circle_radius;

  figure;
  set(gcf, 'Position', [200 200 1000 1000]);
  h = VoxelPlotter(obs, 1);
  h.Vertices = h.Vertices - 0.5;  % Need cells centered at 0.5.
  view(45, 45); hold on; grid on;
  axis vis3d;
  axis([0, dims(1), 0, dims(2), 0, dims(3)])
  grid minor;
  
  % Case 0: pick a random starting point and random displacement.
%   pos = unifrnd(0, 1, 1, 3) .* [width - 1, height - 1, depth - 1];
%   delta = unifrnd(-1/2, 1/2, 1, 3) .* ...
%     [width - 1, height - 1, depth - 1];
%   CalcLineTraceSamplePlot(pos, delta, dims, obs);
  
  % Also plot some hard cases.
  % Case 1: Shouldn't intersect.
%   pos = [5.2, 6.3, 7.4] + 1;
%   delta = unifrnd(-5, 0, 1, 3);
%   [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
%   assert(collide == 0);
%   assert(norm(new_pos - (pos + delta)) < 1e-6);
  
  % Case 2: Should collide with positive border (check them all).
  for dim = 1:3
    pos = dims / 2;
    pos(dim) = dims(dim) - 4.1; % Step back 4.1 units from the border.
    delta = [0, 0, 0];
    delta(dim) = 6.1;  % This should step us past the border.
    expected_pos = pos;
    expected_pos(dim) = dims(dim);  % Should hit the boundary.
    [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
    assert(collide == 1);
    assert(norm(new_pos - expected_pos) < 1e-4);
  end
  % Case 2: Should collide with negative border (check them all).
  for dim = 1:3
    pos = dims / 2;
    pos(dim) = +4.1; % Step forward 4.1 units from the border.
    delta = [0, 0, 0];
    delta(dim) = -6.1;  % This should step us past the border.
    expected_pos = pos;
    expected_pos(dim) = 0;  % Should hit the boundary.
    [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
    assert(collide == 1);
    assert(norm(new_pos - expected_pos) < 1e-4);
  end
  
  % Case 3: Step off all borders (there are lots of cases here, but we'll
  % just try one of them) and make sure we handle this case properly.
  pos = [width - 5.2, height - 6.3, depth - 7.4];
  delta = unifrnd(0, 10, 1, 3) + 10;
  [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
  assert(collide == 1);
  % One of the coordinates should be on the border.
  assert(abs(new_pos(1) - (width)) < 1e-4 || ...
    abs(new_pos(2) - (height)) < 1e-4 || ...
    abs(new_pos(3) - (depth)) < 1e-4);
  
  % Case 4: Step off exactly the corner (just do one of them).
  pos = [width - 1.5, height - 1.5, depth - 1.5];
  delta = [2, 2, 2];
  expected_pos = dims;
  [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
  assert(collide == 1);
  assert(norm(new_pos - expected_pos) < 1e-4);
  
  % Case 4: Step off exactly a mixed corner (just do one of them).
  pos = [width - 0.5, 0.5, depth - 0.5];
  delta = [2, -2, 2];
  [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
  assert(collide == 1);
  expected_pos = [width, 0, depth];
  assert(norm(new_pos - expected_pos) < 1e-4);
  
  % Case 5: Do a collision with the geometry.
  pos = [5.5, height - 3.2, 11.1];
  delta = dims / 3 - pos;
  [collide, ~] = CalcLineTraceSamplePlot(pos, delta, dims, obs);
  assert(collide == 1);
end

function [collide, new_pos] = CalcLineTraceSamplePlot(pos, delta, dims, ...
                                                      obs)
  [collide, new_pos] = CalcLineTrace(pos, delta, dims, double(obs));
  
  % Note: cell centers are all 0.5-based. That is (0.5, 0.5, 0.5) is the
  % center of the first cell.
  % Matlab defines (1, 1) as the middle of the first pixel in the upper
  % left.
  % Plot the line trace.
  plot3([pos(1), pos(1) + delta(1)], [pos(2), pos(2) + delta(2)], ...
    [pos(3), pos(3) + delta(3)], '-g', 'LineWidth', 2);
  % Plot the starting position.
  plot3(pos(1), pos(2), pos(3), 'ok', 'LineWidth', 2);
  % Plot the final position.
  if collide == 1
    plot3(new_pos(1), new_pos(2), new_pos(3), 'oc', 'LineWidth', 2);
  else
    plot3(new_pos(1), new_pos(2), new_pos(3), 'ob', 'LineWidth', 2);
  end
end
