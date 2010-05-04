function mm = gen_svm_data( lst )
for i = 1 : length(lst)
    F = lst{i};
    F = F(:);
    % scale data 
    m_min = min( F(:) );
    m_max = max( F(:) );
    F(:) = (F(:) - m_min) / (m_max - m_min) ;
    d(:,i) = F(:);

end
mm = d';



