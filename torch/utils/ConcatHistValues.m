function [ bin_centers, counts ] = ConcatHistValues( hist, min_val, ...
  max_val, nbins_return, min_val_return, max_val_return )
% What we have is a bunch of histograms of a certain nubmer of bins and
% we'd like to decimate them.

hist = hist';
nbins_input = size(hist, 2);

bin_width = (max_val - min_val) / nbins_input;
bin_centers_input = ((1:nbins_input) - 0.5) * bin_width + min_val;

bin_width = (max_val_return - min_val_return) / nbins_return;
bin_centers = ((1:nbins_return) - 0.5) * bin_width + min_val_return;

counts = zeros(1, nbins_return);

for i = 1:nbins_input
  % This is a dumb solution, but accumulate counts.
  total_count = sum(hist(:, i));
  cur_center_val = bin_centers_input(i);
  
  % Find the closests bin.
  [~, icenter] = min(abs(bin_centers - cur_center_val));
  
  % Accumluate the samples.
  counts(icenter) = counts(icenter) + total_count;
end

% We should have accounted for all samples.
assert(sum(counts) == sum(hist(:)));

end

