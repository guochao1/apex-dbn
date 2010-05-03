function lst = load_kyoto_mat(dirname)
lst = dir( dirname + '/*.mat');
for i = [ 1 : length(lst) ]
    clear OL OS OM;
    nm = dirname+'/'+lst(i).name;
    load( nm );  
    lst{i} = OL;   
end
