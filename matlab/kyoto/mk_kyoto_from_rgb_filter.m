function mk_kyoto_from_rgb_filter( fname, lst )
% parse mat file and save them into a binary formatted file first
% designed for Kyoto Dataset, the binary file can be read by a iterator
% of RBM to do training

fo = fopen( fname, 'wb');
% write the size of file
fwrite( fo, [ 0 ] , 'int32');

count = 0;
for i = [ 1 : length(lst) ]
    nm = lst(i).name;
    img = im2double(rgb2gray( imread(nm)));
    [y_max,x_max,z_max] = size( img );

    ff = [-1 0 1];
    grad_x    = imfilter( img ,  ff  ); 
    grad_y    = imfilter( img , -ff' ); 
    
    if z_max > 1
        [g,idx_x] = max( abs(grad_x) , [], 3 );
        grad_x    = grad_x(:,:,idx_x);   

        [g,idx_y] = max( abs(grad_y) , [], 3 );
        grad_y    = grad_y(:,:,idx_y);                   
    end  
           
    fwrite( fo, [x_max,y_max,2] , 'int32'); 
    fwrite( fo, grad_x', 'float32' );
    fwrite( fo, grad_y', 'float32' );
    count = count + 1;
    clear grad_x grad_y nm img g
end

frewind( fo );
fwrite( fo, [ count ] , 'int32');
fclose( fo );

fprintf(1,'%d parsed\n',count);
