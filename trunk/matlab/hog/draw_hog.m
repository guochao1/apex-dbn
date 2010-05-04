function img = draw_hog( F, c_max, b_max, s_max )
% draw feature histogram of F
if nargin == 1
    c_max = 8;
    b_max = 2;
    s_max = 1;
end

[ y_max, x_max, z_max ] = size( F );

img = zeros( (y_max/b_max-1)*s_max*c_max + b_max*c_max  ,... 
             (x_max/b_max-1)*s_max*c_max +b_max*c_max);

pic = zeros( c_max, c_max, z_max );

for z = 0 : z_max-1
   theta = pi/2 + z*pi/z_max;
   if theta > pi
       theta = theta - pi;
   end
   pic( :,:, z+1 ) = draw_arc( c_max, theta );
end

[A,I] = max( F, [], 3 );   
        
for y = 0 : y_max - 1
    for x = 0 : x_max - 1
        by = floor( y / b_max );
        bx = floor( x / b_max );
        iy = by*s_max*c_max + mod( y, b_max) * c_max;
        ix = bx*s_max*c_max + mod( x, b_max) * c_max;
        
        img( iy+1: iy+c_max , ix+1: ix+c_max) ...
            = img( iy+1 : iy+c_max , ix+1:ix+c_max) ...
            + pic(:,:,I(y+1,x+1)) * A(y+1,x+1); 
    end
end      
img = img / max(img(:));


function img = draw_arc( c_max, theta )
img = zeros( c_max, c_max );

if theta <= pi/4 || theta > pi*3/4
    for x = 0 : c_max-1
        % distance to center
        dx = x  - (c_max-1)/2;
        dy = dx * tan( theta );
        y = c_max - floor( dy + (c_max-1)/2 )-1;
        if y>= 0 && y < c_max
           img( y+1, x+1 ) = 1; 
        end
    end
else
    for y = 0 : c_max-1
        % distance to center
        dy = y  - (c_max-1)/2;
        dx = dy / tan( theta );
        x  = c_max-floor( dx + (c_max-1)/2 )-1;
        if x>= 0 && x < c_max
            img( y+1, x+1 ) = 1; 
        end
    end
end
