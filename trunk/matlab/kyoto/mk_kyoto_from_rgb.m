function mk_kyoto_from_rgb( fname, lst )
% parse mat file and save them into a binary formatted file first
% designed for Kyoto Dataset, the binary file can be read by a iterator
% of RBM to do training

fo = fopen( fname, 'wb');
% write the size of file
fwrite( fo, [ 0 ] , 'int32');

count = 0;
for i = [ 1 : length(lst) ]
    nm = lst(i).name;
    A = im2double( rgb2gray( imread(nm) ));
    
    [yy_max,xx_max] = size( A );
    fwrite( fo, [xx_max,yy_max] , 'int32'); 
    fwrite( fo, A', 'float32' );
    count = count + 1;
end

frewind( fo );
fwrite( fo, [ count ] , 'int32');
fclose( fo );

fprintf(1,'%d parsed\n',count);
