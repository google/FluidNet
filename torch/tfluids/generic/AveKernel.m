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

clc;
aveRad = 1;
aveSize = aveRad * 2 + 1;
sigma = 0.4;
mean = aveRad + 1;
[y, x, z] = ndgrid(1:aveSize, 1:aveSize, 1:aveSize);
aveKernel = exp(-((x - mean).^2 + (y - mean).^2 + (z - mean).^2) / ...
  (2 * sigma^2));
aveKernel = aveKernel / sum(aveKernel(:));
% imshow(aveKernel);
format long e

disp('const real ave_kernel[ave_size * ave_size] = {');
for z = 1:aveSize
  for y = 1:aveSize
    str = '    ';
    for x = 1:aveSize
      str = [str, sprintf('%.7e,', aveKernel(x, y, z))];
      if x < aveSize
        str = [str, ' '];
      end
    end
    disp(str);
  end
end
disp('};');
