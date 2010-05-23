function d = gen_svm_data( lst )
for i = 1 : length(lst)
    F = lst{i};
    d(:,i) = F(:);
end



