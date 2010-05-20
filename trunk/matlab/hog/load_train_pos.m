function lst = load_train_pos( dirname )
% load in postive training examples
dl = dir(strcat(dirname,'\*.jpg') );

yy_max = 30;
xx_max = 14;

fprintf( 1, '%d images in all\n', length(dl));

for i = 1 : length(dl)
    img = imread( strcat( strcat(dirname,'\'), dl(i).name ));
    hog = gen_hog( img );
    [y_max,x_max,z_max] = size( hog );
    
    yy  = floor( (y_max - yy_max)/2); 
    xx  = floor( (x_max - xx_max)/2);
    
    % this is specific for current setting
    lst{i} = hog( yy+1:yy+yy_max, xx+1:xx+xx_max, : );     
end

