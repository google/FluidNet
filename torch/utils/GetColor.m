function [ color ] = GetColor( i )

ColorSet = get(gca, 'ColorOrder');
cur_color = i-1; %% index from 0
cur_color = mod(cur_color,length(ColorSet(:,1))); %% modulus the length
cur_color = cur_color+1; %% index from 1
color = ColorSet(cur_color,:);

end
