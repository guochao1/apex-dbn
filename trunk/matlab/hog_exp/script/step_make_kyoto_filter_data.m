clear all;
load models/hog_class_data.mat

crbm_write_kyoto( 'data/feature_train_pos.filter.bin' , crbm_filter_grad(lst_train_pos) );
crbm_write_kyoto( 'data/feature_train_neg.filter.bin' , crbm_filter_grad(lst_train_neg) );
crbm_write_kyoto( 'data/feature_test_pos.filter.bin'  , crbm_filter_grad(lst_test_pos)  );
crbm_write_kyoto( 'data/feature_test_neg.filter.bin'  , crbm_filter_grad(lst_test_neg)  );

