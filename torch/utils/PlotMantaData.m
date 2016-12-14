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

close all; clearvars; clc;

% This script is just to sanity check Manta data.
% First you need to create a data.mat file by running cnn_fluids.lua and
% calling:
% >> _tr:saveSampleToMatlab(_conf, "utils/data.mat", 1, 1)

load('data.mat');

[gradP_y, gradP_x] = gradient(p);

% Also calculate gradient of P using V = Vdiv - gradP
gradP_x_vel = UxDiv - Ux;
gradP_y_vel = UyDiv - Uy;

gradP = cat(3, gradP_x, gradP_y, zeros(size(gradP_x)));
gradP_vel = cat(3, gradP_x_vel, gradP_y_vel, zeros(size(gradP_x_vel)));
maxGradP = max(max(abs(gradP(:))), max(abs(gradP_vel(:))));

figure;
subplot(1, 3, 1);
% My Matlab ignores range argument.
imshow((gradP + maxGradP) / (2 * maxGradP));
title('gradient(p)');
subplot(1, 3, 2);
imshow((gradP_vel + maxGradP) / (2 * maxGradP));
title('Vdiv - V');

err = (gradP_vel - gradP);
maxErr = max(err(:));

disp(['Max error = ', num2str(maxErr)]);

maxFracErr = max(err(:) ./ max(abs(gradP(:)), 1e-6));
disp(['Max fractional error = ', num2str(maxFracErr)]);

subplot(1, 3, 3);
imshow((err + maxErr) / (2 * maxErr));
title('Error');
