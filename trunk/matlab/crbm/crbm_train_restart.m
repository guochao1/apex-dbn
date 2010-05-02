clear all;
x_max = 10;
y_max = 10;
h_max = 24;
v_max = 1;
sigma = 0.2;

W   = randn( [y_max, x_max, h_max, v_max] );
hb  = randn( [1,h_max] );
vb  = randn( [1,v_max] );

d_hb= zeros( size(hb) );
d_vb= zeros( size(vb) );
d_W = zeros( size(W)  ); 

crbm_train_kyoto;

