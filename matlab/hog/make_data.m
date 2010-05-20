

lst_train_pos = load_train_pos('D:\HiBoost\Dataset\real_dataset\TrainPos')
lst_test_pos  = load_train_pos('D:\HiBoost\Dataset\real_dataset\TestPos')

save models/hog_data.mat *

lst_train_neg = load_train_neg('D:\HiBoost\Dataset\real_dataset\TrainNeg',length(lst_train_pos))
lst_test_neg  = load_train_neg('D:\HiBoost\Dataset\real_dataset\TestNeg' ,length(lst_test_pos))

save models/hog_data.mat *

data_train(:, 1:length(lst_train_pos)) = gen_svm_data( lst_train_pos )
data_train(:, length(lst_train_pos)+1: length(lst_train_pos)+length(lst_train_neg) ) = gen_svm_data( lst_train_neg )

data_test(:, 1:length(lst_test_pos))   = gen_svm_data( lst_test_pos )
data_test(:, length(lst_test_pos)+1: length(lst_test_pos)+length(lst_test_neg) ) = gen_svm_data( lst_test_neg )

save models/hog_data_svm.mat data_train data_test
