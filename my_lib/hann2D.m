function kernel = hann2D(n_rows, n_cols)
%% Input checks
% Check whether n_rows and n_cols are positive integer numbers
if n_rows <= 0 || n_cols <= 0 || mod(n_rows,1)~=0 || mod(n_cols,1)~=0
    error('n_rows and n_cols must be positive intergers');
end

%% Kernel creation
% Create the 1D gaussian kernels
rw = gausswin(n_rows);
cw = gausswin(n_cols);
% Compose them to create the 2D one
kernel = rw * cw';
kernel = kernel ./ sum(kernel(:));
