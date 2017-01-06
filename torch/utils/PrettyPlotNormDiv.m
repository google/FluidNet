function [ ] = PrettyPlotNormDiv(stats, legendstrs)
figure;
set(gcf, 'Position', [100 100 500 350]);
set(gca, 'TickLabelInterpreter', 'tex');

assert(length(stats) == length(legendstrs));
for i = 1:length(stats)
  nruns = size(stats{i}, 1);
  meanDiv = mean(stats{i}, 2);
  x = 0:(nruns - 1);
  plot(x, meanDiv, 'Color', GetColor(i), 'LineWidth', 1.5);
  hold on;
end

for i = 1:length(stats)
  nruns = size(stats{i}, 1);
  meanDiv = mean(stats{i}, 2);
  x = 0:(nruns - 1);
  normStd = std(stats{i}, 0, 2);
  d = 16;
  errorbar(x(1:d:end), meanDiv(1:d:end), normStd(1:d:end), 'x', ...
    'Color', GetColor(i), 'LineWidth', 1.5);
end

grid on;
legend(legendstrs, 'FontSize', 12, ...
  'Location', 'NorthWest', 'Interpreter', 'latex');
xlabel('Timestep', 'FontSize', 15, 'Interpreter', 'latex');
ylabel('$$\mathbf{E}\left(||\nabla \cdot \hat{u}||\right)$$', ...
  'FontSize', 15, 'Interpreter', 'latex');
set(gca, 'FontSize', 14);
xlim([x(1), x(end)]);
set(gcf,'color','w');

end

