function F = crbm_gen_feature( img, model )
F = im2double(rgb2gray(img));
for i = 1 : length( model.layer )
    F = crbm_filter( F,model.layer{i} );
    F = crbm_pool  ( F,model.layer{i} );
end