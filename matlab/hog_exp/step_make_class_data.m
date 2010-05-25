% make data for classification

lst_train_pos = load_train_pos('D:\HiBoost\Dataset\real_dataset\TrainPos');
lst_test_pos  = load_train_pos(['D:\HiBoost\Dataset\real_dataset\TestPos']);

fprintf(1,'positive data set loaded\n');

save models/hog_class_data.mat *

lst_train_neg = gen_train_neg('D:\HiBoost\Dataset\real_dataset\TrainNeg',10);

save models/hog_class_data.mat *

fprintf(1,'all data set loaded\n');
