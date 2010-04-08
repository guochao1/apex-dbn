function check_kyoto( fname )
fi = fopen( fname,'rb');

num = fread( fi, 1, 'int32' )

for i = [1:num-1]
  clear OL
  x_max = fread( fi, 1, 'int32' );
  y_max = fread( fi, 1, 'int32' );
  OL = fread( fi, [ x_max, y_max ] , 'float32' );
  OL = OL';
  if( length(OL) > 1 )
    imshow( OL );
    pause;
  end
end

fclose( fi ) ;
