function img = crbm_draw_tensor( W, norm_method, x_num )
if nargin < 2
    norm_method = 'all';
end
if nargin < 3
    x_num = 7;
end

[y_max,x_max,h_max,v_max] = size( W );

if h_max < x_num
    x_num = h_max
end
if v_max > 1
    W = W(:,:,:,1);
end

switch norm_method
  case 'all'
    W = (W - min(W(:))) / (max(W(:))- min(W(:)));
  case 'local'
    for i = 1 : h_max
        W(:,:,i) = (W(:,:,i) - min(min(W(:,:,i)))) / (max(max(W(:,:,i)))- min(min(W(:,:,i))));
    end
end

yy_max = ceil( h_max / x_num );

img = zeros( yy_max*(y_max+1)+1 , x_num*(x_max+1)+1 );

for i = 1 : h_max
    y = floor( (i-1) / x_num );
    x = mod( i-1 , x_num );
    
    img( y*(y_max+1) + 2 : y*(y_max+1)+y_max+1,...
         x*(x_max+1) + 2 : x*(x_max+1)+x_max+1 ) = W(:,:,i);                  
end
