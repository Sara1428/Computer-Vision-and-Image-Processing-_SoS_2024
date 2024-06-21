% Gray-scale Images are basically black and white images
% Each gray scale image can be thought to be made up of 8 binary images
% Now we are going to extract the 8 bit planes of a gray scale image

img1 = imread("pic1.png");
img1d = double(img1);
count = 0;
adding_bit_planes = 0
while count < 8 
    imshow(mod(floor(img1d/pow2(count)),pow2(count+1)));
    new{count+1} = imsave(imshow(mod(floor(img1d/pow2(count)),pow2(count+1))));
    count = count + 1;
    adding_bit_planes = adding_bit_planes + pow2(count-1);
end

% the bit planes are present as bmp images in Modified_Images folder
% they can joined again and guess what ? we get our original image back
bit_im1 = imread("img1_bit_plane1.bmp");
bit_im2 = imread("img1_bit_plane2.bmp");
bit_im3 = imread("img1_bit_plane3.bmp");
bit_im4 = imread("img1_bit_plane4.bmp");
bit_im5 = imread("img1_bit_plane5.bmp");
bit_im6 = imread("img1_bit_plane6.bmp");
bit_im7 = imread("img1_bit_plane7.bmp");
bit_im8 = imread("img1_bit_plane8.bmp");
orig_img =pow2(5)*bit_im6 +pow2(6)*bit_im7+pow2(7)*bit_im8;
imshow(orig_img);

% resizing images
img_half = imresize(img1, 0.5);
imshow(img_half);
img_double = imresize(img1, 2);
imshow(img_double);
img_new_dimn = imresize(img1, [100,100]);
% the image becomes 100 by 100 instead of 255 by 255
% 0 by 0 is same as 255 by 255
imshow(img_new_dimn);
