clear all;

x_max = 10;
y_max = 10;
h_max = 24;
v_max = 1;
sigma = 0.2;

%W   = randn( [y_max, x_max, h_max, v_max] ) * 0.0001;
%hb  = randn( [1,h_max] ) * 0.0;
%vb  = randn( [1,v_max] ) * 0.0 + 0.5;

%d_hb= zeros( size(hb) );
%d_vb= zeros( size(vb) );
%d_W = zeros( size(W)  ); 



load( 'models/crbm_model_4.mat');
epoch = 4;

crbm_train_kyoto;

