function [Y, score] = gen_detection_svm( img, svm_model, stride, S_r, ...
                                         w_y_max, w_x_max,... 
                                         d_y_max, d_x_max, S_s  )
    if nargin <= 2
        stride = 2;
    end
    if nargin <= 3
        S_r = 1.05;
    end
    if nargin <= 4
        w_y_max = 128;
        w_x_max = 64;
    end
    if nargin <= 6
        d_y_max = 30;
        d_x_max = 14;
    end
    if nargin <= 8
        S_s = 1;
    end
    
    
    count = 0;
    [y_max,x_max,z_max] = size(img); 
    
    S_e = min( y_max / w_y_max, x_max / w_x_max );
    num_s = floor( log( S_e / S_s ) / log( S_r )  + 1 ); 
    scale = S_s;
    
    % enumerate the scale
    for i = 1 : num_s    
        hog = gen_hog( imresize( img, 1/scale ) );
        [yy_max,xx_max,zz_max] = size( hog );
        
        % make feature vector
        for y = 1 : stride : yy_max - d_y_max + 1
            for x = 1 : stride : xx_max - d_x_max + 1
                
                % run binary classifier
                F = hog(y:y+d_y_max-1,x:x+d_x_max-1,:);            
                [pre,acc,dvalue] = svmpredict( [0], F(:)' , svm_model );
                
                % push correct prediction to list
                if dvalue(1) > 0
                    count = count + 1;
                    % find detection point
                    if yy_max == d_y_max
                        yy = y_max / 2; 
                    else
                        yy = ((w_y_max/2) + (y-1)/(yy_max-d_y_max)*(y_max/scale - w_y_max ))*scale; 
                    end
                    if xx_max == d_x_max
                        xx = x_max / 2; 
                    else
                    xx = ((w_x_max/2) + (x-1)/(xx_max-d_x_max)*(x_max/scale - w_x_max ))*scale; 
                    end
                    
                    Y( count, 1 ) = xx;
                    Y( count, 2 ) = yy;
                    Y( count, 3 ) = scale;
                    score( 1, count ) = dvalue(1);
                end
            end
        end
        scale = scale * S_r;
    end
    

