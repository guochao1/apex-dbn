function YD = fusion_detection( Y, score, sigma,...
                                lambda,...
                                eps_i, eps_r )
% fusion the dection by finding the local point of gaussian density
% Y        : 3 * m matrix, m is the number of detection   
% score    : 1 * m matrix, m is the number of detection
% sigma    : 3 * 1 matrix, x, y, s
% lambda   : decision threshold


if nargin < 3
    sigma = [ 8, 16, log(1.6) ]';
end
if nargin < 4 
    lambda = 4;
end
if nargin < 5
    eps_i = 0.001;
    eps_r = 8;
end 


[n,m] = size( Y );
Y(3,:)= log( Y(3,:) );

% compute convariance matrix
H = repmat( sigma.^2 , [1,m] );
H(1,:) = H(1,:) .* (exp( Y(3,:) ).^2);
H(2,:) = H(2,:) .* (exp( Y(3,:) ).^2);
H_inv  = 1 ./ H;
H_det  = 1 ./ sqrt(H(1,:).*H(2,:).*H(3,:));

% start find local detection 
count = 0;
for i = 1 : m
    ym = detect_proc( Y, score, H_inv, H_det, Y(:,i), eps_i );
    
    new_rst = 1;
    % find overlap detections
    for j = 1 : count
        if norm(ym - YD(:,j)) <= eps_r
            new_rst = 0;
            ss(:,j) = ss(:,j) + score(:,i);
        end
    end
    if new_rst == 1
        count = count + 1;
        YD( :, count ) = ym;
        ss(:,count) = score(:,i);
    end
end

YD(3,:) = exp( YD(3,:) );
YD = YD(:,ss > mean(score) * lambda );

function ym = detect_proc( Y, score, H_inv, H_det, ym, eps_i )
err = 10;
while err > eps_i
    ymm = detect_one_iter( Y, score, H_inv, H_det, ym );
    err = norm(ym - ymm);
    ym  = ymm; 
end

% one iteration of decttion
function ym = detect_one_iter( Y, score, H_inv , H_det, ym )  
[n,m] = size( Y );
% compute gaussian weight
smooth_w = score .* exp( - 0.5* sum((Y-repmat(ym,[1,m])).^2 .* H_inv)).*H_det;
smooth_w = smooth_w / sum( smooth_w );

Hh_inv = H_inv * smooth_w';

% compute next iteration by formula
ym  = ( 1./Hh_inv ) .* ( ( H_inv .* Y ) * smooth_w' );  

