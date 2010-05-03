learning_rate = 0.01;
momentum      = 0.5;
wd            = 0.01;

sparse_level  = 0.003;
sparse_lambda = 5.0;

num_iter = 10; 

lst = load_kyoto_mat('kyoto'); 

for itr = [ epoch+1 : epoch + num_iter ]
    for i = [ 1 : length(lst) ]
        [dd_W,dd_hb,dd_vb,dd_sps,errt] = crbm_cal_update( lst{i},...
                                                          W, hb, vb,...
                                                          sigma,sparse_level);
        fprintf( 1, '%d,%d err=%f\n' , itr, i, errt);
        
        d_W  = d_W *momentum + dd_W  - wd*learning_rate*W;   
        d_hb = d_hb*momentum + dd_hb + dd_sps;
        d_vb = d_vb*momentum + dd_vb;
        
        W = W  + d_W;
        hb= hb + d_hb; 
        vb= vb + d_vb;     
    end 
    model_name = strcat('models/crbm_model_', num2str( itr) );
    save( model_name, 'W', 'd_W', 'hb', 'd_hb', 'vb', 'd_vb', 'sigma');
end


