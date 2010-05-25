function model = load_crbm_model( fname )
% utils to load crbm model from file
fid = fopen( fname, 'rb');

count = fread( fid, 1, 'uint32');
fread( fid, 1, 'uint32');

if count < 0
    error('error loading CRBN model');
end

for i = 1:count
    model.layer{i}.model_type = fread( fid, 1, 'int32');
    model.layer{i}.pool_size  = fread( fid, 1, 'int32');
    model.layer{i}.v_max      = fread( fid, 1, 'int32');
    model.layer{i}.h_max      = fread( fid, 1, 'int32');
    model.layer{i}.y_max      = fread( fid, 1, 'int32');
    model.layer{i}.x_max      = fread( fid, 1, 'int32');
    model.layer{i}.v_sigma    = fread( fid, 1, 'float32');
    model.layer{i}.h_sigma    = fread( fid, 1, 'float32');
    model.layer{i}.v_bias_prior_mean= fread( fid, 1, 'float32');
    model.layer{i}.h_bias_prior_mean= fread( fid, 1, 'float32');
    model.layer{i}.v_init_sigma     = fread( fid, 1, 'float32');
    model.layer{i}.h_init_sigma     = fread( fid, 1, 'float32');
    model.layer{i}.w_init_sigma     = fread( fid, 1, 'float32');

    model.layer{i}.h_bias   = load_tensor_1D( fid ); 
    model.layer{i}.v_bias   = load_tensor_1D( fid ); 
    model.layer{i}.W        = load_tensor_4D( fid ); 
    model.layer{i}.d_h_bias = load_tensor_1D( fid ); 
    model.layer{i}.d_v_bias = load_tensor_1D( fid ); 
    model.layer{i}.d_W      = load_tensor_4D( fid );     
end
fclose( fid );

function v = load_tensor_1D( fid )
x_max = fread( fid, 1, 'int32' );
v = fread(fid, x_max, 'float32');
function W = load_tensor_4D( fid )
x_max = fread( fid, 1, 'int32' );
y_max = fread( fid, 1, 'int32' );
h_max = fread( fid, 1, 'int32' );
v_max = fread( fid, 1, 'int32' );
for v = 1 : v_max
    for h = 1 : h_max 
        W(:,:,h,v) = fread(fid, [x_max,y_max], 'float32')';
    end
end

 