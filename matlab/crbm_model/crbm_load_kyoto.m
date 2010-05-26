function lst = crbm_load_kyoto( fname, F )

fi = fopen( fname,'rb');

num = fread( fi, 1, 'int32' );

for i =  1 : num 
    
    x_max = fread( fi, 1, 'int32' );
    y_max = fread( fi, 1, 'int32' );
    z_max = fread( fi, 1, 'int32' );
    for z = 1 : z_max
        A(:,:,z) = fread( fi, [ x_max, y_max  ] , 'float32' )';
    end
    lst{i} = A;    
end

fclose( fi ) ;

