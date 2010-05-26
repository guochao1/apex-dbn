% default parameter for HOG
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



fname_model = '';
fname_img   = '';



load(fname_model);
img = imread( fname_img );

[Y,score] = gen_detection_hog_linear( img, df_param, linear_model.w, ...
                                      0 );

fprintf(1,'detection generated\n');

% get positive detection
imgd = draw_detection( img, YY(:,score>0) );

YD   = fusion_detection( Y(:,score>0), score(:,score>0) ); 

imgf = draw_detection( img, YD );


 
