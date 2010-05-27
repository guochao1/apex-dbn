function dlst = gen_detection_list_crbm_filter_linear( dirname, fnamelist,...
                                                       crbm_model, ...
                                                       B, bias, c ...
                                                      )
if nargin < 6
    c = 0;
end

% generate list of detections for certain image
%
[d_y_max,d_x_max] = crbm_get_feature_size( 128,64, crbm_model );
for i = 1 : length( fnamelist ) 
    fname = strcat( dirname, fnamelist(i).name );
    img = imread( fname );
    [Y,score] = gen_detection_crbm_filter_linear( img, crbm_model,...
                                                  B  , bias,...
                                                  d_y_max, d_x_max );
    dlst{i}.Y = Y( :, score > c );
    dlst{i}.score = score( :, score>c );
    fprintf(1,'%d:%s\n', i, fnamelist(i).name );   
end
