function check_kyoto( fname )
fi = fopen( fname,'rb');

x_max = fread( fi, 1, 'int32' )
y_max = fread( fi, 1, 'int32' )
z_max = fread( fi, 1, 'int32' )

for i = [1:z_max-1]
  clear OL
  OL = fread( fi, [ x_max, y_max ] , 'float32' );
  OL = OL';
  if( length(OL) > 1 )
    imshow( OL );
    pause;
  end
end

fclose( fi ) ;
