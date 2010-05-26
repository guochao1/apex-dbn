clear all;
load models/hog_class_data.mat

crbm_write_kyoto( 'data/feature_train_pos.bin' , lst_train_pos );
crbm_write_kyoto( 'data/feature_train_neg.bin' , lst_train_neg );
crbm_write_kyoto( 'data/feature_test_pos.bin'  , lst_test_pos  );
crbm_write_kyoto( 'data/feature_test_neg.bin'  , lst_test_neg  );

