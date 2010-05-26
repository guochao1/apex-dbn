function [Y, score] = gen_detection_crbm_linear( img    , param,...
                                                 B      , bias,...
                                                 d_y_max, d_x_max,...
                                                 stride , S_r, ...
                                                 w_y_max, w_x_max,... 
                                                 S_s  )
% generate detection using HOG and linear model,
% 
% img     : input image
% param   : parameter for feature extraction
%
% B       : linear classification weight
% bias    : linear classification bias
%
% d_y_max 
% d_x_max : shape of classification window in feature space
%
% stride  : stride in target feature space
% S_r     : stride in scale
%
% w_y_max  
% w_x_max : shape of classification window in pixel 
%
% S_s     : start scale


    if nargin < 5
        d_y_max = 39;
        d_x_max = 18;
    end
    
    if nargin < 7
        stride = 2
    end
    if nargin < 8
        S_r = 1.05;
    end
    if nargin < 9
        w_y_max = 128;
        w_x_max = 64;
    end
    if nargin < 11
        S_s = 1;
    end
    
    count = 0;    
    [y_max,x_max,z_max] = size(img); 
    
    S_e = min( y_max / w_y_max, x_max / w_x_max );
    num_s = floor( log( S_e / S_s ) / log( S_r )  + 1 ); 
    scale = S_s;
    
    % enumerate the scale
    for i = 1 : num_s    
        cf = crbm_gen_feature( imresize( img, 1/scale ), param );
        [yy_max,xx_max,zz_max] = size( cf );
        
        % make feature vector
        for y = 1 : stride : yy_max - d_y_max + 1
            for x = 1 : stride : xx_max - d_x_max + 1
                                % run binary classifier
                F = cf(y:y+d_y_max-1,x:x+d_x_max-1,:);            
                dvalue = B * F(:) + bias;
                
                % push prediction to list
                count = count+1;
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
                
                Y( 1, count ) = xx;
                Y( 2, count ) = yy;
                Y( 3, count ) = scale;
                score( 1, count ) = dvalue;
            end
        end
        scale = scale * S_r;
    end
    

