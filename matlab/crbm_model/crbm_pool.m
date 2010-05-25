function Wout = crbm_pool( Win, layer )
% filter the data to next layer
[y_max,x_max,z_max] = size( Win );
if z_max ~= layer.h_max
    error('h_max unmatch');
end
pool_size = layer.pool_size;
yy_max = floor(y_max /layer.pool_size);
xx_max = floor(x_max /layer.pool_size);

for z = 1 : z_max
    for y = 1 : pool_size: y_max
        for x = 1: pool_size : x_max
            WW = Win(y:y+pool_size-1,x:x+pool_size-1,z);
            Wout(ceil(y/pool_size),ceil(x/pool_size),z) = sum( WW(:) );
        end
    end
end







