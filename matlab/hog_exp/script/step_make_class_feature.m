% make feature for classification experiment 
clear all;
load models/hog_class_data.mat


% default parameter 
df_param.c_y_max = 8;
df_param.c_x_max = 8;
df_param.b_y_max = 2;
df_param.b_x_max = 2;
df_param.num_hist= 9;
df_param.slide_step = 1;
df_param.signed = 'unparam.signed';
df_param.norm_method = 'l1sqrt';
df_param.w_y_max = 30;
df_param.w_x_max = 14;

% specify parameters here
param = df_param;
fname = 'models/feature_class_df';
set_name ='default setting';
substep_make_class_feature;


 

