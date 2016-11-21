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

function [] = PlotEpochs()
% A simple utility to plot the per-epoch performance of a bunch of
% models (so you can compare learning rates, hyperparams, etc). This is
% very hacky.

global prev_models;
clc;

% Less permissive selection:
path = '../../data/models';

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

disp('Found model files:');
disp(models);
disp('In path:');
disp(path);
disp(' ');

handles = cell(1,6);
Plot(path, models, handles);

end

function [] = Plot(path, models, handles)
data = cell(1, length(models));

% Get the results.
for i = 1:length(models)
  cur_model = models{i};
  
  disp(['Processing ', cur_model]);
  cur_data = readtable([path, '/', cur_model], 'Delimiter', '\t');
  cur_data = cur_data(:, 1:end-1);
  data{i} = cur_data;
  
  if i == 1
    header = cur_data.Properties.VariableNames;
  else
    cur_header = cur_data.Properties.VariableNames;
    for j = 1:length(cur_header)
      assert(strcmp(header{j}, cur_header{j}), ...
        'cant mix models with different output types');
    end
  end
end


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
  
  % Plot each of the columns specified by indices.
  for j = 1:length(indices)
    y = cur_data{:, indices(j)};
    x = 1:length(y);
    if logy && ~logx
      semilogy(x, y, line_spec, 'LineWidth', 2, ...
        'Color', GetColor((i - 1) * length(indices) + j)); hold on;
    elseif ~logy && logx
      semilogx(x, y, line_spec, 'LineWidth', 2, ...
        'Color', GetColor((i - 1) * length(indices) + j)); hold on;
    elseif logy && logx
      loglog(x, y, line_spec, 'LineWidth', 2, ...
        'Color', GetColor((i - 1) * length(indices) + j)); hold on;      
      
    else
      plot(x, y, line_spec, 'LineWidth', 2, ...
        'Color', GetColor((i - 1) * length(indices) + j)); hold on;          
    end
    xstart = min(xstart, x(1));
    xend = max(xend, x(end));
    ystart = min(ystart, min(y));
    yend = max(yend, y(1));
    legend_str{length(legend_str) + 1} = ...
      [header{indices(j)}, ': ', model_str];
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
legend(legend_str, 'Location', 'NorthEast', 'FontSize', ...
  7, 'Interpreter', 'none');    
xlabel(xlabel_str, 'FontSize', 14);
ylabel(ylabel_str, 'FontSize', 14);
end
