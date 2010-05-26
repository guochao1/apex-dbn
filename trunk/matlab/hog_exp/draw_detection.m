function img = draw_detection( img, Y, w_y_max, w_x_max )
%DRAW_DETECTION Summary of this function goes here
%   Detailed explanation goes here
if nargin <= 2
    w_y_max = 128;
    w_x_max = 64;
end
[n,m] = size(Y);

for i = 1 : m
    x = Y( 1, i );
    y = Y( 2, i );
    s = Y( 3, i );     
    x_a = x - w_x_max/(s*2);
    x_b = x + w_x_max/(s*2);
    y_a = y - w_y_max/(s*2);
    y_b = y + w_y_max/(s*2);
    img = draw_line_x( img, y_a , x_a, x_b );
    img = draw_line_x( img, y_b , x_a, x_b );
    img = draw_line_y( img, x_a , y_a, y_b );
    img = draw_line_y( img, x_b , y_a, y_b );    
end

function img = draw_line_x( img, y, x_s, x_e )
    y   = floor( y );
    x_s = floor( x_s );
    x_e = floor( x_e );
    
    [y_max,x_max,z_max] = size( img );
    
    if y <= y_max && y > 0
       for x = max(1,x_s) : min( x_max, x_e )
           img( y, x, : ) = 0;
           img( y, x, 1 ) = 255;
       end
    end



function img = draw_line_y( img, x, y_s, y_e )
    x   = floor( x );
    y_s = floor( y_s );
    y_e = floor( y_e );
    [y_max,x_max,z_max] = size( img );
    if x <= x_max && x > 0
       for y = max(1,y_s) : min( y_max, y_e )
           img( y, x, : ) = 0;
           img( y, x, 1 ) = 255;
       end
    end

