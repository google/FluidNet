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

function [ ] = test_spatial_finite_elements( )

clc;

M = zeros(5, 5);
M(:, :) = ...
  [[1.0000  0.6250  0.5000  0.6250  1.0000]; ...
   [0.6250  0.2500  0.1250  0.2500  0.6250]; ...
   [0.5000  0.1250  0.0000  0.1250  0.5000]; ...
   [0.6250  0.2500  0.1250  0.2500  0.6250]; ...
   [1.0000  0.6250  0.5000  0.6250  1.0000]];

display(M);

[dfdx, dfdy] = gradient(M);
display(dfdx);
display(dfdy);

end

