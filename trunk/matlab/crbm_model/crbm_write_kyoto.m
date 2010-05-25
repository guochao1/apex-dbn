function crbm_write_kyoto( fname, lst )
% save list of pic into kyoto format

fo = fopen( fname, 'wb');
% write the size of file
fwrite( fo, [ 0 ] , 'int32');

count = 0;
for i = [ 1 : length(lst) ]
    A = im2double( rgb2gray( imread(lst{i}) ));
    
    [yy_max,xx_max] = size( A );
    fwrite( fo, [xx_max,yy_max,1] , 'int32'); 
    fwrite( fo, A', 'float32' );
    count = count + 1;
end

frewind( fo );
fwrite( fo, [ count ] , 'int32');
fclose( fo );

fprintf(1,'%d parsed\n',count);

