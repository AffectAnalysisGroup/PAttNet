

img = imread('temp_image.png');
mkdir('patches');
imshow(img)
img_patch1 = imcrop(img,[131 125 125 125]); 
img_patch1 = imresize(img_patch1, [100 100]);
imwrite(img_patch1, ['patches/patch1.png']);

img_patch2 = imcrop(img,[256 125 125 125]); 
img_patch2 = imresize(img_patch2, [100 100]);
imwrite(img_patch2, ['patches/patch2.png']);

img_patch3 = imcrop(img,[131 160 125 125]); 
img_patch3 = imresize(img_patch3, [100 100]);
imwrite(img_patch3, ['patches/patch3.png']);

img_patch4 = imcrop(img,[193 160 125 125]); 
img_patch4 = imresize(img_patch4, [100 100]);
imwrite(img_patch4, ['patches/patch4.png']);

img_patch5 = imcrop(img,[256 160 125 125]);
img_patch5 = imresize(img_patch5, [100 100]);
imwrite(img_patch5, ['patches/patch5.png']);

img_patch6 = imcrop(img,[131 250 125 125]);
img_patch6 = imresize(img_patch6, [100 100]);
imwrite(img_patch6, ['patches/patch6.png']);

img_patch7 = imcrop(img,[193 250 125 125]); 
img_patch7 = imresize(img_patch7, [100 100]);
imwrite(img_patch7, ['patches/patch7.png']);

img_patch8 = imcrop(img,[256 250 125 125]); 
img_patch8 = imresize(img_patch8, [100 100]);
imwrite(img_patch8, ['patches/patch8.png']);

img_patch9 = imcrop(img,[193 300 125 125]); 
img_patch9 = imresize(img_patch9, [100 100]);
imwrite(img_patch9, ['patches/patch9.png']);




