%example code to convert all the jpg files in current directory to
%binary file

fname = 'INDRIA.train_pos.kyoto.bin'
lst   = dir('*.jpg')

mk_kyoto_from_rgb( fname, lst )  

%you can check the file by
check_kyoto( fname )