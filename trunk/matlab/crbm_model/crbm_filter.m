function Wout = crbm_filter( Win, layer )
% filter the data by one layer of crbm
[y_max,x_max,v_max] = size( Win );
if v_max ~= layer.v_max
    error('v_max unmatch');
end
h_max = layer.h_max;
yy_max = floor((y_max - layer.y_max+1)/layer.pool_size)*layer.pool_size;
xx_max = floor((x_max - layer.x_max+1)/layer.pool_size)*layer.pool_size;
y_max  = yy_max+layer.y_max-1;
x_max  = xx_max+layer.x_max-1;


for h = 1 : h_max
    Wout(:,:,h) = ones( yy_max,xx_max ) * layer.h_bias(h); 
end

for h = 1 : h_max
    for v = 1 : v_max
        Wout(:,:,h) = Wout(:,:,h) + conv2( Win(1:y_max,1:x_max,v) , layer.W(end:-1:1,end:-1:1,h,v),'valid');
    end
end

if layer.model_type == 2
    Wout = Wout ./ (layer.v_sigma^2);
end
Wout = norm_maxpooling( Wout, layer.pool_size );

function Wout = norm_maxpooling( Win, pool_size )
[y_max,x_max,z_max] = size( Win );
for z = 1 : z_max
    for y = 1 : pool_size: y_max
        for x = 1: pool_size : x_max
            WW = Win(y:y+pool_size-1,x:x+pool_size-1,z);
            t  = max( WW(:) ); 
            WW(:) = exp( WW(:)-t );
            s  = sum( WW(:) );
            WW(:) = WW(:) / ( s + exp(-t) );            
            Wout(y:y+pool_size-1,x:x+pool_size-1,z) = WW;
        end
    end
end



