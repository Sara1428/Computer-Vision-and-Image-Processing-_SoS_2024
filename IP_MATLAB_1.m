% Learning and implementing Image Processing tools of MATLAB
% BASICS 

% reading an image
img1 = imread('pic1.png');
% displaying an image
imshow(img1);
% displaying the information of the image
disp(imfinfo("pic1.png"));

% the given image is of colotype 'truecolor', which is basically a rgb type image

% in order to see information pixel-wise, we can use impixelinfo

impixelinfo(imshow(img1));

% img2 is a 255 by 255 black image
img2 = uint8(zeros(255,255));
imshow(img2);

% img3 is a 255 by 255 white image
img3 = uint8(255 * ones(255,255));
imshow(img3);

% We can interconvert images of different color types using built-in
% functions of MATLAB
[x, xmap] = rgb2ind(img1,3,"nodither");
imshow(x,xmap);

% modifying the image img1 using some property extraction
c = uint8(255* (img1>150));
imshow(c);
