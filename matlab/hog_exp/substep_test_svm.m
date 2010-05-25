load(fname_feature);
load(fname_model);

data_test = [ feature_test_pos, feature_test_neg];

num_test  = size( data_test ); 
label_test = ones( 1, num_test ) * (-1);
label_test( 1:size( feature_test_pos, 2 ) ) = 1;

fprintf( 1, 'data ready, start testing...');

[pred,acc,dvalue] = svmpredict( label_test' , data_test' , svm_model ); 

[ffpw,miss] = cal_DET( label_test, dvalue );

loglog(  ffpw, miss, ps );
hold on;

