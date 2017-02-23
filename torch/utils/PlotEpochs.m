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

% @param path - optional path for model data.
% @param path - optional cell of filenames to plot.
function [] = PlotEpochs(models)
% A simple utility to plot the per-epoch performance of a bunch of
% models (so you can compare learning rates, hyperparams, etc). This is
% very hacky.

global prev_models;

% Less permissive selection:
if nargin < 1
  clc;
  
  % User will select models.
  path = '../../data/models/2017_02_13/';

  files = dir([path, '/*_log.txt']);
  files = {files.name};
  assert(length(files) > 0, 'No models to plot!');
  files = sort(files);
  ok_files = files;

  [s, ok] = listdlg('PromptString', ...
    'Select all log files to plot ("cancel" = plot the old set)',...
    'SelectionMode', 'multiple', 'ListString', ok_files, ...
    'ListSize', [1200, 800]);

  if ~ok
    models = prev_models;
    if isempty(models)
      error('No previous models to print');
    end
  else
    models = cell(1, length(s));
    for i = 1:length(s)
      models{i} = ok_files{s(i)};
    end
    prev_models = models;
  end

  if ~iscell(models)
    models = {models};
  end
  
  for i = 1:length(models)
    models{i} = [path, '/', models{i}];
  end
else
  assert(iscell(models));
end

disp('Found model files:');
disp(models);

handles = cell(1,6);
Plot(models, handles);

end

function [] = Plot(models, handles)
data = {};

% Get the results.
valid_models = [];
for i = 1:length(models)
  cur_model = models{i};
  
  disp(['Processing ', cur_model]);
  cur_data = readtable( cur_model, 'Delimiter', '\t');
  if (size(cur_data, 1) == 0)
    disp(['WARNING **** No data in ', cur_model, ' skipping ****']);
  else
    cur_data = cur_data(:, 1:end-1);
    data{length(data) + 1} = cur_data;
    valid_models(length(valid_models) + 1) = i;

    if length(data) == 1
      header = cur_data.Properties.VariableNames;
    else
      cur_header = cur_data.Properties.VariableNames;
      for j = 1:length(cur_header)
        assert(strcmp(header{j}, cur_header{j}), ...
          'cant mix models with different output types');
      end
    end
  end
end
assert(length(valid_models) >= 1, 'No models with data!');
models = models(valid_models);

% Choose the headers to plot.
headers = data{1}.Properties.VariableNames;
[indices, ok] = listdlg('PromptString', ...
  'Select the values to plot ("cancel" = ["teLoss", "trLoss"])',...
  'SelectionMode', 'multiple', 'ListString', headers, ...
  'ListSize', [500, 300]);

if length(indices) == 0 || ~(ok)
  % The default is 'trLoss' and 'teLoss'.
  indices = [find(ismember(headers, 'trLoss')), ...
    find(ismember(headers, 'teLoss'))];
end

% Plot the results
NicePlot(data, indices, models, 'epochs', 'total loss', true, true);
end

function [color] = GetColor(i)
% ColorSet = get(gca, 'ColorOrder');
ColorSet = [         0,         0,         1; ...
                     0,       0.5,         0; ...
                     1,         0,         0; ...
                     0,      0.75,      0.75; ...
                  0.75,         0,      0.75; ...
                  0.75,      0.75,         0; ...
                  0.25,      0.25,      0.25; ...
                     0,         1,         0; ...
                   0.5,         0,         0; ...
                     1,         1,         0; ...
                     0,         1,         1; ...
                     0,       0.5,         1];
cur_color = i-1; %% index from 0
cur_color = mod(cur_color,length(ColorSet(:,1))); %% modulus the length
cur_color = cur_color+1; %% index from 1
color = ColorSet(cur_color,:);
end

function [] = NicePlot(data, indices, models, xlabel_str, ylabel_str, ...
  logx, logy)
assert(length(models) == length(data));
assert(length(indices) > 0);
   
legend_str = {};
legend_handles = [];

smoothing_window = 3;
raw_thickness = 1.5;
raw_alpha = 0.1;
smooth_thickness = 2;

figure;
set(gcf, 'Position', [100 100 600 800]);
best_model = zeros(1, length(indices));
best_model_val = ones(1, length(indices)) * inf;
% line_types = {'-', '--', ':', '-.'};
% line_types = {'--', '-'};
line_types = {'-'};
xstart = inf;
xend = -inf;
ystart = inf;
yend = -inf;
for i = 1:length(models)
  model_str = models{i};
  cur_data = data{i};
  header = cur_data.Properties.VariableNames;
  line_spec = line_types{mod(i - 1, length(line_types)) + 1};
  
  % Plot each of the columns specified by indices. Plot both the raw and
  % smoothed versions.
  for j = 1:length(indices)
    y = cur_data{:, indices(j)};
    x = 1:length(y);
    
    % Firstly, we might have done a restart with different a different
    % loss. This would look like 2 segments, with a sudden jump. In this
    % case, lets just use the second segment.
    %
    % Unfortunately, this code will not be robust because a suddent jump
    % might also mean instability or dynamic LR change. Therefore, we
    % should print out to the user that we're taking a sub-set.
    dy = abs(diff(y));
    % Normalize the delta.
    dy = dy / std(dy);
    ijump = find(dy > 5, 1, 'first') + 1;  % +1 because of dy length.
    if (~isempty(ijump))
      disp(['WARNING: Restart detected at epoch ', ijump, ' for model:']);
      disp(model_str);
      x = x(ijump:end);
      y = y(ijump:end);
    end
    
    
    cur_win = max(min(smoothing_window, ceil(length(y - 1) / 3)), 2);
    if logy && ~logx
      lraw = semilogy(x, y, line_spec, 'LineWidth', raw_thickness, ...
        'Color', GetColor((i - 1) * length(indices) + j));
      hold on;
      [xsmooth, ysmooth] = smooth(x, log(y), cur_win);
      ysmooth = exp(ysmooth);
      lsmooth = semilogy(xsmooth, ysmooth, line_spec, 'LineWidth', ...
        smooth_thickness, 'Color', ...
        GetColor((i - 1) * length(indices) + j));
    elseif ~logy && logx
      lraw = semilogx(x, y, line_spec, 'LineWidth', raw_thickness, ...
        'Color', GetColor((i - 1) * length(indices) + j));
      hold on;
      [xsmooth, ysmooth] = smooth(log(x), y, cur_win);
      xsmooth = exp(xsmooth);
      lsmooth = semilogx(xsmooth, ysmooth, line_spec, 'LineWidth', ...
        smooth_thickness, 'Color', ...
        GetColor((i - 1) * length(indices) + j));
    elseif logy && logx
      lraw = loglog(x, y, line_spec, 'LineWidth', raw_thickness, ...
        'Color', GetColor((i - 1) * length(indices) + j));     
      hold on;
      [xsmooth, ysmooth] = smooth(log(x), log(y), cur_win);
      xsmooth = exp(xsmooth);
      ysmooth = exp(ysmooth);
      lsmooth = loglog(xsmooth, ysmooth, line_spec, 'LineWidth', ...
        smooth_thickness, 'Color', ...
        GetColor((i - 1) * length(indices) + j));
    else
      lraw = plot(x, y, line_spec, 'LineWidth', raw_thickness, ...
        'Color', GetColor((i - 1) * length(indices) + j)); hold on;     
      hold on;
      [xsmooth, ysmooth] = smooth(x, y, cur_win);
      lsmooth = plot(xsmooth, ysmooth, line_spec, 'LineWidth', ...
        smooth_thickness, 'Color', ...
        GetColor((i - 1) * length(indices) + j));
    end
    
    lraw.Color = [lraw.Color(1), lraw.Color(2), lraw.Color(3), raw_alpha];
    
    xstart = min(xstart, x(1));
    xend = max(xend, x(end));
    ystart = min(ystart, min(y));
    yend = max(yend, max(y(:)));
    
    % For the legend we don't want the full path.
    str_decomp = strsplit(model_str, '/');
    legend_str{length(legend_str) + 1} = ...
      [header{indices(j)}, ': ', str_decomp{end}];
    legend_handles(length(legend_handles) + 1) = lsmooth;
    if (best_model_val(j) > min(y))
      best_model_val(j) = min(y);
      best_model(j) = i;
    end
  end
end
axis([xstart xend ystart yend]);
for j = 1:length(indices)
  disp(['Best model for ', header{indices(j)} ' is ', ...
    models{best_model(j)}, ' (min val = ', num2str(best_model_val(j)), ...
    ')']);
end
grid on;
legend(legend_handles, legend_str, 'Location', 'NorthEast', 'FontSize', ...
  8, 'Interpreter', 'none');    
xlabel(xlabel_str, 'FontSize', 14);
ylabel(ylabel_str, 'FontSize', 14);
end

function [xsmooth, ysmooth] = smooth(x, y, win)
% We will use filtfilt to filter the data because I don't have the correct
% toolbox for the 'smooth' Matlab function.
% recall: filtfilt does forward and backward passes to prevent FIR phase
% delay

% Firstly, resample (x, y) to even intervals (this is our first hack).
assert(all(diff(x) > 0));  % X must be monotonicly increasing.
[ysmooth, xsmooth] = resample(y, x);

% filtfilt, has a property that it forces the final response to
% exactly intersect the end points (rather than their average), which is
% undesirable. We therefore add a fictitious data point on each end that is
% the interpolation of a locally fit line, i.e. we continue the trend of
% the last win values and use this as the end point for filtering.

% The Forier transform of a M length moving average is:
% H[f] = sin(pi * f * M) / M * sin(pi * f)
% --> The 3db point is pretty ugly and figuring out it's phase response
% at this point is not all that meaningful anyway (it's far from linear
% phase). Just use the last n points and call it a day.
nfit = win;

pstart = polyfit(xsmooth(1:nfit), ysmooth(1:nfit)', 1);
xstart = xsmooth(1) - abs(xsmooth(2) - xsmooth(1));
ystart = pstart(1) * xstart + pstart(2);

pend = polyfit(xsmooth(end - nfit + 1:end), ...
  ysmooth(end - nfit + 1:end)', 1);
xend = xsmooth(end) + abs(xsmooth(end) - xsmooth(end - 1));
yend = pend(1) * xend + pend(2);

ysmooth = [ystart; ysmooth; yend];

ysmooth = filtfilt(ones(win, 1) / (win), 1, ...
  ysmooth);

% Now remove the fictitious end points.
ysmooth = ysmooth(2:end - 1);
end