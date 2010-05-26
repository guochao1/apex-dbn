function F = crbm_gen_feature( img, model )

[y_max,x_max,z_max] = size(img);

if z == 3
    F = im2double(rgb2gray(img));
else
    F = img;
end

for i = 1 : length( model.layer )
    F = crbm_filter( F,model.layer{i} );
    F = crbm_pool  ( F,model.layer{i} );
end
