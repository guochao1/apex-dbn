function [B,bias] = logistic_reg_train( G, R, ...
                                        GG, RR,...
                                        learning_rate, wd, momentum,...
                                        num_iter, ...
                                        B, bias )
% train a linear model using logistic regression
% G: v * h matrix, with h columns of global-featur
% R: 1 * h matrix, true answer of prediction
%
% B            : 1 * v matrix global factor
% bias         : global bias

if nargin < 4
    learning_rate = 0.01;
    wd = 1.0;
    momentum = 0.5; 
end
if nargin < 8
    num_iter = 100;
end

[v,h] = size( G );

if nagin < 10
    B = randn( 1, n ) * 0.0001;
    bias = 0;
end

dB    = zeros( 1, v );
dbias = 0;

for iter = 1 : num_iter    
    W = 1./(1 + exp( - B*G -bias ));    
    E = R - W;

    if mod( iter, plot_step ) == 0
        idx     = iter / plot_step;
        xx(idx) = iter;
        yy(idx) = mean( abs(R -(W>0.5))); 
        yyt(idx)= mean( abs(RR-((1./(1+exp(-B*GG-bias))) >0.5)));
        plot(xx,yy);
        hold;
        plot(xx,yyt,'red');
        xlabel('iter');            
        ylabel('Acc' );
        drawnow;
    end
    % calculate gradient
    dB    = dB +  (E * G') / h - wd * B;
    dbias = dbias + sum( E )/h;
    
    % update weight
    B = B + learning_rate * dB;
    bias = bias + learning_rate * dbias;

    dB = dB * momentum;
    dbias = dbias * momentum; 

    dbias
    nm_dB = norm( dB )
end





