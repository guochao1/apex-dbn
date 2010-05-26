clear all;
load models/hog_class_data.mat

fname = 'models/feature_crbm';



crbm_model = crbm_load_model('');

fprintf(1,'generate feature for %s train_pos\n',fname);
feature_train_pos = gen_crbm_feature( lst_train_pos, crbm_model );
fprintf(1,'generate feature for %s train_neg\n',fname);
feature_train_neg = gen_crbm_feature( lst_train_neg, crbm_model );
fprintf(1,'generate feature for %s test_pos\n',fname);
feature_test_pos = gen_crbm_feature( lst_test_pos, crbm_model );
fprintf(1,'generate feature for %s test_neg\n',fname);
feature_test_neg = gen_crbm_feature( lst_test_neg, crbm_model );

save( fname, 'feature_train_pos', 'feature_train_neg',...
      'feature_test_pos', 'feature_test_neg', 'crbm_model');    
    
