function lst = load_train_pos( dirname )
% load in postive training examples
dl = dir(strcat(dirname,'\*.jpg') );
fprintf( 1, '%d images in all\n', length(dl));

for i = 1 : length(dl)
    lst{i} = imread( strcat( strcat(dirname,'\'), dl(i).name ));
end
