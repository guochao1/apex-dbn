function F = gen_crbm_kyoto_feature( lst, yy_max, xx_max, F )
% generate hog feature in n*h matrix
% n: number of feature, 
% h: number of data 
if nargin < 4
    F = single([]);
end

[n,h] = size( F );

% specific setting for 1 layer
if nargin < 2
    yy_max = 39;
    xx_max = 18;
end
 

for i = 1: length( lst )
    cf = lst{i};
    [y_max,x_max,z_max] = size( cf );

    yy  = floor( (y_max - yy_max)/2); 
    xx  = floor( (x_max - xx_max)/2);
    
    % this is specific for current setting
    FF = cf( yy+1:yy+yy_max, xx+1:xx+xx_max, : );     
    F(:,i+h) = single(FF( : ));
end
