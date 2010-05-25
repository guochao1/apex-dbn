function [y_max,x_max] = crbm_get_feature_size( y_max, x_max, model )
for i = 1 : length( model.layer )
    ly = model.layer{i};
    y_max = y_max - ly.y_max+1;
    x_max = x_max - ly.x_max+1;
    y_max = floor( y_max / ly.pool_size);
    x_max = floor( x_max / ly.pool_size);
end
