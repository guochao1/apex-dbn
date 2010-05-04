function lst = load_train_pos( dirname )
% load in postive training examples
dl = dir(strcat(dirname,'/*.png') );

fprintf( 1, '%d images in all\n', length(dl));

for i = 1 : length(dl)
    img = imread( strcat( strcat(dirname,'/'), dl(i).name ));
    hog = gen_hog( img );
    % this is specific for current setting
    lst{i} = hog( 5:4+30 , 5:4+14, : );     
end

