clear all;
fname = 'models/feature_crbm_filter_00';
crbm_model = crbm_load_model( 'D:\User\tqchen\data_set\dbn_model\00.inria_filter.model' );
[y_max,x_max] = crbm_get_feature_size( 128, 64, crbm_model )

fprintf(1,'generate feature for %s train_pos\n',fname);
feature_train_pos = gen_crbm_kyoto_feature(crbm_load_kyoto('feature/crbm_filter_00_train_pos.bin'),y_max,x_max);
fprintf(1,'generate feature for %s train_neg\n',fname);
feature_train_neg = gen_crbm_kyoto_feature(crbm_load_kyoto('feature/crbm_filter_00_train_neg.bin'),y_max,x_max);
fprintf(1,'generate feature for %s test_pos\n',fname);
feature_test_pos = gen_crbm_kyoto_feature(crbm_load_kyoto('feature/crbm_filter_00_test_pos.bin'),y_max,x_max);
fprintf(1,'generate feature for %s test_neg\n',fname);
feature_test_neg = gen_crbm_kyoto_feature(crbm_load_kyoto('feature/crbm_filter_00_test_neg.bin'),y_max,x_max);

save( fname, 'feature_train_pos', 'feature_train_neg',...
             'feature_test_pos', 'feature_test_neg');    
    

         