learning_rate = 0.01;

momentum      = 0.5;
wd            = 0.01;

sparse_level  = 0.003;
sparse_lambda = 5.0;

num_iter = 10; 


lst = dir( '*.mat');

for itr = [ 1 : num_iter ]
    for i = [ 1 : length(lst) ]
        nm = lst(i).name;
        load( nm );  
        [dd_W,dd_hb,dd_vb,dd_sps,errt] = crbm_cal_update( OL,...
                                                          W, hb, vb,...
                                                          sigma,sparse_level);
        fprintf( 1, '%d,%d err=%f\n' , itr, i, errt);
        
        d_W  = d_W *momentum + dd_W - wd*learning_rate*W;   
        d_hb = d_hb*momentum + dd_hb + dd_sps;
        d_vb = d_vb*momentum + dd_vb;
        
        W = W  + d_W;
        hb= hb + d_hb; 
        vb= vb + d_vb; 
        clear OL OS OM;
    end
    
    save crbm_model.mat  W d_W hb d_hb vb d_vb sigma;
end

