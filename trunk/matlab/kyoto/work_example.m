%example code to convert all the jpg files in current directory to
%binary file

fname = 'INDRIA.train_pos.kyoto.bin'
y_max = 160
x_max = 96
lst   = dir('*.jpg')

mk_kyoto_from_rgb( fname, y_max, x_max, lst )  

%you can check the file by
check_kyoto( fname )