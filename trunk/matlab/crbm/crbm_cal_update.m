function [d_W,d_hb,d_vb,d_sps,errt,sps]=crbm_cal_update( v_pos, ...
                                                     W, hb, vb,...
                                                     sigma, sparse_level)
    
[y_max ,x_max,h_max,v_max] = size( W );
[v_y_max,v_x_max,t ]       = size( v_pos );
h_y_max = v_y_max - y_max + 1;
h_x_max = v_x_max - x_max + 1;

% calculate h_pos
for hh = 1 : h_max
    h_pos(:,:,hh) = ones( h_y_max, h_x_max  )*hb( hh );
end
for vv = 1 : v_max
    for hh = 1 : h_max
        h_pos(:,:,hh) = h_pos(:,:,hh) + conv2(v_pos(:,:,vv),W(end:-1:1,end:-1:1,hh ,vv),'valid');
    end
end
% calculate mean value
h_pos = 1 ./ ( 1 + exp( - h_pos ./ (sigma*sigma) ) );

% sample h_pos
h_neg = double( h_pos > rand( size(h_pos ) ));

% go down, sample v
for vv = 1 : v_max
    v_neg(:,:,vv) = ones( v_y_max, v_x_max)*vb( vv );
end

for vv = 1 : v_max
    for hh = 1 : h_max
        v_neg(:,:,vv) = v_neg(:,:,vv) + conv2(h_neg(:,:,hh),W(:,:,hh ,vv),'full');
    end
end

% calculate err and sps
sps  = mean(h_pos(:));
errt = mean((v_pos(:)-v_neg(:)).^2);

%sample v_neg
v_neg = v_neg + randn( size(v_neg) )*sigma;

% up again
for hh = 1 : h_max
    h_neg(:,:,hh) = ones( h_y_max, h_x_max  )*hb( hh );
end
for vv = 1 : v_max
    for hh = 1 : h_max
        h_neg(:,:,hh) = h_neg(:,:,hh) + conv2(v_neg(:,:,vv),W(end:-1:1,end:-1:1,hh ,vv),'valid');
    end
end
h_neg = 1 ./ ( 1 + exp(- h_neg./(sigma*sigma) ) );

% calculate update 
d_W = zeros( size(W) );

for vv = 1 : v_max
    for hh = 1 : h_max
        for x = 1 : x_max
            for y = 1 : y_max
                d_W(y,x,hh,vv) =...
                    sum(sum( v_pos(y:y+h_y_max-1,x:x+h_x_max-1,vv).* h_pos(:,:,hh) ))...
                    - sum(sum( v_neg(y:y+h_y_max-1,x:x+h_x_max-1,vv).* h_neg(:,:,hh))); 
            end
        end
    end
end

for vv = 1 : v_max
    d_vb(vv) = sum(sum( v_pos(:,:,vv))) - sum(sum(v_neg(:,:,vv)));
end

h_size =  h_y_max * h_x_max; 
v_size =  v_y_max * v_x_max;

for hh = 1 : h_max    
    d_hb (hh) =  sum(sum(h_pos(:,:,hh))) - sum(sum(h_neg(:,:,hh)));
    % sparse regularization
    d_sps(hh) = -(sum(sum(h_pos(:,:,hh)))/h_size  - sparse_level)* ...
        sum(sum((h_pos(:,:,hh).*(1-h_pos(:,:,hh)))))./ h_size;
end
d_W  = d_W ./ h_size;
d_hb = d_hb./ h_size;
d_vb = d_vb./ v_size;

