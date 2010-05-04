function lst = load_train_neg( dirname, num_sample, s_y_max, s_x_max )
if nargin == 2
    s_y_max = 30;
    s_x_max = 14;
end
% load in postive training examples
dl = dir(strcat(dirname,'/*.png') );

num_image      = length(dl);

for i = 1 : num_sample
    i
    idx = floor( rand() * num_image ) + 1;
    
    img = imread( strcat( strcat(dirname,'/'), dl(idx).name ));
    hog = gen_hog( img );

    % random extract hog 
    [y_max,x_max,z_max] = size( hog );    
    y = floor ( rand() * (y_max - s_y_max) );
    x = floor ( rand() * (x_max - s_x_max) );
     
    % this is specific for current setting
    lst{i} = hog( y+1:y+s_y_max , x+1:x+s_x_max, : );     
end

