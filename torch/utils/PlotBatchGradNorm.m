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

clearvars;

model = 'clip_disabled';
% model = 'clip_1';
modeldir = '../../data/models/';

filepreamble = [modeldir, '/', model, '_batchGradNorm_Epoch'];
files = dir([filepreamble, '*.mat']);
files = {files.name};
assert(length(files) > 0, 'No epochs to plot!');

batchGradNormVals = {};
epochs = [];
for i = 1:length(files)
  load([modeldir, '/', files{i}]);
  % result is in batchGradNorm and epoch.
  batchGradNormVals{i} = batchGradNorm;
  epochs(i) = epoch;
end

% Plot a histogram of the first and last epoch gradients.
% for i = [1, length(batchGradNormVals)]
for i = [1, 227]
  batchGradNorm = batchGradNormVals{i};
  figure;
  set(gcf, 'Position', [200 200 1200 600]);
  subplot(1, 2, 1);
  plot(batchGradNorm); grid on;
  title(['L2 gradient for each batch (epoch ', num2str(epochs(i)), ...
    ', model: ', model, ')'], 'Interpreter', 'none');
  xlabel('Batch number'); ylabel('L2 of gradient');
  subplot(1, 2, 2);
  hist(batchGradNorm, 40); grid on;
  xlabel('L2 of gradient'); ylabel('Count');
  title(sprintf('mean: %f, std: %f', mean(batchGradNorm), std(batchGradNorm)));
end

% Plot the mean and std over all epochs.
meanVals = zeros(length(batchGradNormVals), 1);
stdVals = zeros(length(batchGradNormVals), 1);
for i = 1:length(batchGradNormVals)
  meanVals(i) = mean(batchGradNormVals{i});
  stdVals(i) = std(batchGradNormVals{i});
end
figure;
plot(epochs, meanVals, 'r'); hold on; grid on;
plot(epochs, stdVals, 'b');
xlabel('Epoch number');
legend({'mean(norm(grad))', 'std(norm(grad))'}, 'Location', 'SouthWest');
title(['Model: ', model], 'Interpreter', 'none');
