function [B,bias] = logistic_reg_train( G, R, ...
                                        GG, RR,...
                                        learning_rate, wd, momentum,...
                                        num_iter, plot_step, ...
                                        B, bias )
% train a linear model using logistic regression
% G: v * h matrix, with h columns of global-featur
% R: 1 * h matrix, true answer of prediction
%
% B            : 1 * v matrix global factor
% bias         : global bias

if nargin < 5
    learning_rate = 0.001;
    wd = 0.0;
    momentum = 0.0; 
end
if nargin < 8
    num_iter = 10000;
    plot_step= 10;
end

[v,h] = size( G );

if nargin < 10
    B = randn( 1, v ) * 0.01;
    bias = 0;
end

% scale first
G (:, R<0)  = G(:,R<0) * (-1);  
GG(:, RR<0) = GG(:,RR<0) * (-1);  

dB    = zeros( 1, v );
dbias = 0;

for iter = 1 : num_iter    
    W = 1./(1 + exp( - B*G -bias ));    

    if mod( iter, plot_step ) == 1
        idx     = ceil(iter / plot_step);        
        xx(idx) = iter;
        yy(idx) = mean( log( W ));       
        yyt(idx)= mean( log( 1./(1+exp(-B*GG-bias) )));
        plot(xx,yy);
        hold on;
        plot(xx,yyt,'red');
        xlabel('iter');            
        ylabel('likelihood' );
        drawnow;
        
        err_train = mean( W > 0.5 )
        err_test  = mean(  1./(1+exp(-B*GG-bias) ) > 0.5 )

        dbias
        nm_dB = norm( dB )
    end
    % calculate gradient
    dB    = dB +  (E * G') / h - wd * B/h;
    dbias = dbias + sum( E )/h;
    
    % update weight
    B = B + learning_rate * dB;
    bias = bias + learning_rate * dbias;

    dB = dB * momentum;
    dbias = dbias * momentum; 

end





