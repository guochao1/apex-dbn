load(fname_feature);
load(fname_model);

data_test = [ feature_test_pos, feature_test_neg];

num_test  = size( data_test,2 ); 
label_test = ones( 1, num_test ) * (-1);
label_test( 1:size( feature_test_pos, 2 ) ) = 1;

clear feature*;

fprintf( 1, 'data ready, start testing...\n');

[pred,acc,dvalue] = predict( label_test' , sparse(double(data_test))' , linear_model ); 

[ffpw,miss] = cal_DET( label_test, dvalue );

loglog(  ffpw, miss, ps );
hold on;

