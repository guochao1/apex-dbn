load(fname_feature);

data_train = [ feature_train_pos, feature_train_neg];

num_train  = size( data_train ); 
label_train = ones( 1, num_train ) * (-1);
label_train( 1:size( feature_train_pos, 2 ) ) = 1;

fprintf( 1, 'data ready, start training...');
svm_model = svm_model = svmtrain( label_train', data_train', '-t 0' )
save( fname_model, 'svm_model');



