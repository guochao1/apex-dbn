model = load_crbm_model('00.kyoto3.modify.model');
img   = im2double( rgb2gray(imread(['/home/crow/ML_exp/data_set/' ...
                    'INRIAPerson/Test/pos/crop001504.png'])));


Wou   = crbm_filter( img, model.layer{1} );
Wp    = crbm_pool  ( Wou, model.layer{1} );
 
igw   = draw_tensor( Wp,'local' );

imshow( igw );
