fprintf(1,'generate feature for %s train_pos\n',fname);
feature_train_pos = gen_hog_feature( lst_train_pos, param );
fprintf(1,'generate feature for %s train_neg\n',fname);
feature_train_neg = gen_hog_feature( lst_train_neg, param );
fprintf(1,'generate feature for %s test_pos\n',fname);
feature_test_pos = gen_hog_feature( lst_test_pos, param );

save( fname, 'feature_train_pos', 'feature_train_neg',...
      'feature_test_pos', 'param', 'set_name');    
    
