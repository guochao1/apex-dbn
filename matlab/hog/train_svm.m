clear all;
load models/hog_data_svm.mat

num_train = size( data_train, 2 );
label_train = ones( 1, num_train ) * (-1);
label_train( 1:floor(num_train/2) ) = 1;

num_test = size( data_test, 2 );
label_test = ones( 1,num_test ) * (-1);
label_test( 1:floor(num_test/2) ) = 1;

fprintf( 1, 'data ready, start training...');
svm_model = svmtrain( label_train', data_train', '-t 0' )

save models/hog_svm_model.mat svm_model;
fprintf( 1, 'training model, save model');

test_svm;


