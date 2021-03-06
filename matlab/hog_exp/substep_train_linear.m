load(fname_feature);

data_train = [ feature_train_pos, feature_train_neg];

num_train  = size( data_train, 2 ); 
label_train = ones( 1, num_train ) * (-1);
label_train( 1:size( feature_train_pos, 2 ) ) = 1;

clear feature*;

fprintf( 1, 'data ready, start training...\n');
linear_model = train( label_train', sparse(double(data_train))', '-s 0' );
save( fname_model, 'linear_model');



