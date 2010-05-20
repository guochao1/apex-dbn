function F = gen_hog( img, c_y_max, c_x_max,... 
                      b_y_max, b_x_max, num_hist, ...
                      slide_step, signed, norm_method )
% generate HOG feature given input image
% img: input image
% size of cell and block, slide_step specifies the slide step of
% block

if nargin < 2
    % set default parameters value.
    c_y_max = 8;
    c_x_max = 8;
    b_y_max = 2;
    b_x_max = 2;
    num_hist= 9;
    slide_step = 1;
    signed = 'unsigned';
    norm_method = 'l1sqrt';
else
    if nargin < 10
        error('Input parameters are not enough.');
    end
end
[ y_max, x_max, z_max ] = size( img );

s_y_max = floor((floor( y_max / c_y_max ) - b_y_max)/ slide_step);
s_x_max = floor((floor( x_max / c_x_max ) - b_x_max)/ slide_step);

yy_max  =  (s_y_max * slide_step + b_y_max ) * c_y_max;
xx_max  =  (s_x_max * slide_step + b_x_max ) * c_x_max;
 
ff = [-1 0 1];

grad_x    = imfilter( im2double(img) ,  ff  ); 
grad_y    = imfilter( im2double(img) , -ff' ); 
if z_max > 1
    grad_x = max( grad_x , [], 3 );
    grad_y = max( grad_y , [], 3 );
end  
 
if yy_max ~= y_max || xx_max ~= x_max
    %fprintf( 1, 'warning:image size does fit all the cell size\n');
    yy = floor( (y_max-yy_max)/2 );
    xx = floor( (x_max-xx_max)/2 );
    grad_x = grad_x( yy+1:yy+yy_max, xx+1:xx+xx_max);
    grad_y = grad_y( yy+1:yy+yy_max, xx+1:xx+xx_max);
    y_max  = yy_max;
    x_max  = xx_max; 
end

% length of the arch
grad_r    = sqrt(grad_x.*grad_x + grad_y.*grad_y );
% calculate rotation of the arch
grad_theta= atan2( grad_x, grad_y );

if strcmp( signed, 'unsigned' ) == 1
    grad_theta( grad_theta < 0 )   = grad_theta( grad_theta < 0 ) + pi;
    c_z_max = pi/num_hist;
else
    grad_theta = grad_theta + pi;
    c_z_max = 2*pi/num_hist;
end

% calculate number of blocks in each dimension
b_y_num = ( y_max / c_y_max - b_y_max ) / slide_step+1;
b_x_num = ( x_max / c_x_max - b_x_max ) / slide_step+1;

% generate bins for histogram
F  = zeros( b_y_num*b_y_max, b_x_num*b_x_max, num_hist );
sF = zeros( b_y_max        , b_x_max        , num_hist ); 

% slide to generate blocks
for by = 0 : b_y_num - 1
    for bx = 0 : b_x_num - 1 
       y_s = by*slide_step*c_y_max+1 : (by*slide_step + b_y_max) *c_y_max;
       x_s = bx*slide_step*c_x_max+1 : (bx*slide_step + b_x_max) *c_x_max;
       blk_r     = reweight_gaussian( grad_r(y_s, x_s) );
       blk_theta = grad_theta( y_s, x_s );

       % now we have block weight, we need to vote into histogram       
       for y = 0 : c_y_max*b_y_max - 1
           for x = 0 : c_x_max*b_x_max - 1
               r     = blk_r( y+1, x+1 );
               theta = blk_theta( y+1, x+1 );

               if r == 0
                   continue;
               end
               
               % generate bin index , note the bin index may be
               % invalid on the boundary
               biny1 = floor( (y-(c_y_max-1)/2) / c_y_max ) + 1;
               binx1 = floor( (x-(c_x_max-1)/2) / c_x_max ) + 1;
               binz1 = floor( theta / c_z_max ) + 1;
               if binz1 > num_hist
                  binz1 = 1;
               end
               biny2 = biny1 + 1; 
               binx2 = binx1 + 1; 
               binz2 = binz1 + 1;                
               if binz2 > num_hist
                   binz2 = 1;
               end

               z1  = (binz1-1)*c_z_max;
               y1  = (biny1-1)*c_y_max  + (c_y_max-1)/2;
               x1  = (binx1-1)*c_x_max  + (c_x_max-1)/2;

               % trilinear 
               if binx1 > 0 
                   if biny1 > 0
                       sF( biny1, binx1, binz1 ) =  sF( biny1, binx1, binz1 )...
                           + r*( 1-(theta-z1)/c_z_max )*( 1- (y-y1)/c_y_max )*(1-(x-x1)/c_x_max);

                       sF( biny1, binx1, binz2 ) =  sF( biny1, binx1, binz2 )...
                           + r*( (theta-z1)/c_z_max   )*( 1- (y-y1)/c_y_max )*(1-(x-x1)/c_x_max);
                   end
                   if biny2 <= b_y_max
                       sF( biny2, binx1, binz1 ) =  sF( biny2, binx1, binz1 )...
                           + r*( 1-(theta-z1)/c_z_max )*( (y-y1)/c_y_max )*(1-(x-x1)/c_x_max);

                       sF( biny2, binx1, binz2 ) =  sF( biny2, binx1, binz2 )...
                           + r*( (theta-z1)/c_z_max   )*( (y-y1)/c_y_max )*(1-(x-x1)/c_x_max);                      
                   end
               end                              

               if binx2 <= b_x_max 
                   if biny1 > 0
                       sF( biny1, binx2, binz1 ) =  sF( biny1, binx2, binz1 )...
                           + r*( 1-(theta-z1)/c_z_max )*( 1- (y-y1)/c_y_max )*((x-x1)/c_x_max);

                       sF( biny1, binx2, binz2 ) =  sF( biny1, binx2, binz2 )...
                           + r*( (theta-z1)/c_z_max   )*( 1- (y-y1)/c_y_max )*((x-x1)/c_x_max);
                   end
                   if biny2 <= b_y_max
                       sF( biny2, binx2, binz1 ) =  sF( biny2, binx2, binz1 )...
                           + r*( 1-(theta-z1)/c_z_max )*( (y-y1)/c_y_max )*((x-x1)/c_x_max);

                       sF( biny2, binx2, binz2 ) =  sF( biny2, binx2, binz2 )...
                           + r*( (theta-z1)/c_z_max   )*( (y-y1)/c_y_max )*((x-x1)/c_x_max);                      
                   end
               end                                                            

           end
       end      
       sF( sF<0 ) = 0;            
       % normalize the block histogram 
       eps = 1e-3;
       switch norm_method
           case 'none'
           case 'l1'
               sF(:) = sF(:) / ( norm(sF(:),1) + eps ); 
           case 'l2'
               sF(:) = sF(:) / sqrt( norm(sF(:),2)^2 + eps*eps ); 
           case 'l1sqrt'
               sF(:) = sqrt( sF(:) / ( norm(sF(:),1) + eps )); 
           otherwise
               error('norm method not supported');
       end

       % now we have stored sF, we need to store them
       % into global storage      
       F( by*b_y_max+1 : by*b_y_max+b_y_max,...
          bx*b_x_max+1 : bx*b_x_max+b_x_max,...
          : ) = sF;
       sF(:) = 0;
    end        
end


function F = reweight_gaussian( img )
    [y_max,x_max] = size(img);
    sigma = x_max * 0.5;
    [gaussy, gaussx] = meshgrid(0:y_max-1, 0:x_max-1);
    weight = exp(-((gaussx-(x_max-1)/2).^2+(gaussy-(y_max-1)/2).^2) /(2*sigma*sigma) );
    F = img.*weight;


