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

clearvars; close all;

% A quick sanity check.
simga = 1;
npnts = 1000;
scale = 0.2028 + abs(normrnd(0, 1, npnts) * simga);
disp(mean(scale(:)));
histogram(scale, 40);
