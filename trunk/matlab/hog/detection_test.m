clear all;
load models/hog_svm_model_detect.mat;
img = imread( 'D:\HiBoost\Dataset\real_dataset\TestDetect\0000.jpg');
[Y,score] = gen_detection_svm( img, svm_model )
YD = detection_fusion_gaussian( Y, score )
imgx = draw_detection( img, YD );
imshow( imgx );