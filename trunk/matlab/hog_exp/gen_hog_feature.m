function F = gen_hog_feature( lst, param, F )
% generate hog feature in n*h matrix
% n: number of feature, 
% h: number of data 

if nargin <3
    F = single([]);
end

[n,h] = size( F );

yy_max = param.w_y_max;
xx_max = param.w_x_max;

for i = 1: length( lst )
    hog = gen_hog( lst{i}, param );
    [y_max,x_max,z_max] = size( hog );
    
    yy  = floor( (y_max - yy_max)/2); 
    xx  = floor( (x_max - xx_max)/2);
    
    % this is specific for current setting
    FF = hog( yy+1:yy+yy_max, xx+1:xx+xx_max, : );     
    F(:,i+h) = single(FF( : ));
end
