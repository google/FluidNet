clearvars; close all; clc;

trdata = load('../../data/models/yang_lr0.00025_relu_teStats.mat');
tedata = load('../../data/models/yang_lr0.00025_relu_trStats.mat');

nruns = size(trdata.normDiv, 1);
assert(size(tedata.normDiv, 1) == nruns);  % Sanity check that.

figure;
set(gcf, 'Position', [100 100 500 350]);
set(gca, 'TickLabelInterpreter', 'tex');
d = 1;  % Decimate so it doesn't look too crowded.
plot(0:(nruns - 1), trdata.normDiv(:, 1:d:end), 'LineWidth', 1.5);
grid on;
xlabel('Timestep', 'FontSize', 15, 'Interpreter', 'latex');
ylabel('$$||\nabla \cdot \hat{u}||$$', 'FontSize', 15, 'Interpreter', ...
  'latex');
set(gca, 'FontSize', 14);
% xlim([0, nruns - 1]);
% ylim([0, 20]);
set(gcf,'color','w');

PrettyPlotNormDiv({trdata.normDiv, tedata.normDiv}, ...
  {'Test-set', 'Training-set'});

nbins = 50;

% Also plot some error histograms.
data = tedata;
[centers, counts] = ConcatHistValues(data.pTargetHist, ...
  data.pTargetMin, data.pTargetMax, nbins, -1, 1);
PrettyPlotHist(centers, counts, 'p Target');

[centers, counts] = ConcatHistValues(data.pErrHist,  ...
  data.pErrMin, data.pErrMax, nbins, -1, 1);
PrettyPlotHist(centers, counts, 'p Error');

[centers, counts] = ConcatHistValues(data.UTargetNormHist,  ...
  data.UTargetNormMin, data.UTargetNormMax, nbins, 0, 100);
PrettyPlotHist(centers, counts, '|U Target|');

[centers, counts] = ConcatHistValues(data.UErrNormHist,  ...
  data.UErrNormMin, data.UErrNormMax, nbins, 0, 50);
PrettyPlotHist(centers, counts, '|U Error|');

[centers, counts] = ConcatHistValues(data.divHist,  ...
  data.divMin, data.divMax, nbins, 0, 50);
PrettyPlotHist(centers, counts, 'div(U Predicted)');


