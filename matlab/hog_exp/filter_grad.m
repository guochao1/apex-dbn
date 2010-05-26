function [ lou ] = filter_grad( lst )
for i = 1 : length(lst)
    img = im2double( rgb2gray(lst{i}));
    ff  = [ -1 0 1 ];
    F(:,:,1) = imfilter( img,   ff );
    F(:,:,2) = imfilter( img, -ff' );
    lou{i} = F;
end
