load(fname_feature);

data_train = [ feature_train_pos, feature_train_neg ];
num_train  = size( data_train, 2 ); 
label_train= zeros( 1, num_train ) ;
label_train( 1:size( feature_train_pos, 2 ) ) = 1;

data_test = [ feature_test_pos, feature_test_neg ];
num_test  = size( data_test, 2 ); 
label_test= zeros( 1, num_train ) ;
label_test( 1:size( feature_test_pos, 2 ) ) = 1;

clear feature*;

fprintf( 1, 'data ready, start training...');
[lB, lbias] = log_reg_train( data_train, label_train, data_test, label_test );
save( fname_model, 'lB', 'lbias');



