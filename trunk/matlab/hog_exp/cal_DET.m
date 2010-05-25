function [ffpw,miss] = cal_DET( label_test, dvalue )
[vv,idx] = sort( dvalue );
sol = label_test';
sol = sol( idx );
pos_sol = sol ==  1;
neg_sol = sol == -1;
n = length(sol);

for i = 1 : n
    false_pos = sum( pos_sol(1:i-1) );
    true_pos  = sum( pos_sol(i:n) );
    false_neg = sum( neg_sol(i:n) );

    ftpw(i)   = false_pos / n;
    miss(i)   = false_neg / (true_pos+false_neg);    
end


