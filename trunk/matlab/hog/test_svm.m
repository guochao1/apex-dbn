clear all;
load models/hog_data_svm.mat
load models/hog_svm_model.mat

num_test = size( data_test, 2 );
label_test = ones( 1,num_test ) * (-1);
label_test( 1:floor(num_test/2) ) = 1;

fprintf( 1, 'data ready, start testing...');

[pred,acc,dvalue] = svmpredict( label_test' , data_test' , svm_model )

% producing Precision recall curve
[vv,idx] = sort( dvalue, 'descend' );
sol = label_test';
sol = sol( idx );
hit = 0;
num_pos = sum( sol > 0 );
for i = 1 : length(sol)
    hit = hit + (sol(i)+1)/2;
    pre(i) = hit / i;
    rec(i) = hit / num_pos;
end

plot( rec, pre );

xlabel('recall');
ylabel('precision');





