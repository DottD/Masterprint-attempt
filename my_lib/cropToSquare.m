function img = cropToSquare(img)
%% Crop image to a square with side equal to the shortest dimension
ms = min(size(img));
ds = (size(img)-ms)/2;
img = img(1+floor(ds(1)):end-ceil(ds(1)), 1+floor(ds(2)):end-ceil(ds(2)) );