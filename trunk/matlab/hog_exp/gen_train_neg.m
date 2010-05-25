function lst = gen_train_neg( dirname, num_sample_per_img, s_y_max, s_x_max )
% load in postive training examples
if nargin < 4
    s_y_max = 134;
    s_x_max = 70;
end

dl = dir(strcat(dirname,'\*.jpg') );

for i = 1 :  length(dl)
    img = imread( strcat( strcat(dirname,'\'), dl(i).name ));
    % random extract region
    [y_max,x_max,z_max] = size( img );    
    
    for ii = 1 : num_sample_per_img
        y = floor ( rand() * (y_max - s_y_max) );
        x = floor ( rand() * (x_max - s_x_max) );
        % this is specific for current setting
        lst{ (i-1)*num_sample_per_img + ii } = img( y+1:y+s_y_max , x+1:x+s_x_max, : );     
    end
end

