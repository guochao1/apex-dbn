learning_rate = 0.01;
momentum      = 0.5;
wd            = 0.01;

sparse_level  = 0.003;
sparse_lambda = 5.0;

num_iter = 300; 

lst = load_kyoto_mat('D:\User\tqchen\data_set\kyoto'); 

for itr = [ epoch+1 : epoch + num_iter ]
    
    for i = [ 1 : length(lst) ]
        [dd_W,dd_hb,dd_vb,dd_sps,errt,sps] = crbm_cal_update( lst{i},...
                                                              W, hb, vb,...
                                                              sigma,sparse_level);
        fprintf( 1, '%d,%d err=%f,sps=%f\n' , itr, i, errt,sps);
        err_l( i ) = errt;
        sps_l( i ) = sps;
        
        d_W  = d_W *momentum + dd_W  - wd*W;   
        d_hb = d_hb*momentum + dd_hb + sparse_lambda * dd_sps;
        d_vb = d_vb*momentum + dd_vb;
        
        W = W  + learning_rate * d_W;
        hb= hb + learning_rate * d_hb; 
        vb= vb + learning_rate * d_vb;     
    end 
    model_name = strcat('models/crbm_model_', num2str( itr ) );
    save( model_name, 'W', 'd_W', 'hb', 'd_hb', 'vb', 'd_vb', 'sigma');
    fprintf( 1, '%d epoch end, avg_err=%f,avg_sps=%f\n', itr, mean(err_l), mean(sps_l) );
    
    ee( itr - epoch ) = mean(err_l);
    ii( itr - epoch ) = itr - epoch;
end



