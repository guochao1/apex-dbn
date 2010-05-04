function dl = load_kyoto_mat(dirname)
lst = dir( strcat(dirname , '\*.mat'));
for i = [ 1 : length(lst) ]
    clear OL OS OM;
    nm = strcat( strcat(dirname,'\'),lst(i).name );
    load( nm );  
    dl{i} = OL;   
end
fprintf( 1, '%d loaded\n', length(dl));
