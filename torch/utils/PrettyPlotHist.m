function [ ] = PrettyPlotNormDiv(centers, counts, nameStr)
figure;
set(gcf, 'Position', [100 100 500 350]);
set(gca, 'TickLabelInterpreter', 'tex');

bar(centers, counts, 'LineWidth', 2);

grid on;
xlabel(nameStr, 'FontSize', 15, 'Interpreter', 'latex');
ylabel('Count', 'FontSize', 15, 'Interpreter', 'latex');
set(gca, 'FontSize', 14);
set(gcf,'color','w');

end

