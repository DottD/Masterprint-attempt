function kernel = gauss2D(n_rows, n_cols, varargin)
%% Gauss2D Documentation
% This function creates a bidimensional gaussian filter with dimensions
% given by n_rows and n_cols. 
% The syntax is
%
% kernel = gauss2D(n_rows, n_cols)
%
% or
%
% kernel = gauss2D(n_rows, n_cols, alpha_rows, alpha_cols)
%
% where in the last form alpha_rows stands for the gaussian opening along
% rows, and alpha_cols stands for the gaussian opening along cols.
% The returned kernel is normalized with respect to the sum of its
% elements.

%% Input checks
% Check whether n_rows and n_cols are positive integer numbers
if n_rows <= 0 || n_cols <= 0 || mod(n_rows,1)~=0 || mod(n_cols,1)~=0
    error('n_rows and n_cols must be positive intergers');
end
% Read input alpha specifications, if any
n_opt_argin = nargin-2;
if n_opt_argin >= 1
    % Read the first alpha spec, i.e. for n_rows
    alpha_rows = varargin{1};
    if n_opt_argin >= 2
        % Read the second alpha spec, i.e. for n_cols
        alpha_cols = varargin{2};
        % Last check: warns the user if he used too many arguments
        if n_opt_argin >= 3
            warning('Too many arguments in gauss2D');
        end
    else 
        % Assign default value to the second alpha value
        alpha_cols = floor(n_cols/2)/2;
    end
else
    % Assign default value to both alpha values
    alpha_rows = floor(n_rows/2)/2;
    alpha_cols = floor(n_cols/2)/2;
end

%% Kernel creation
% Create the 1D gaussian kernels
rw = gausswin(n_rows, n_rows/(2*alpha_rows) );
cw = gausswin(n_cols, n_cols/(2*alpha_cols) );
% Compose them to create the 2D one
kernel = rw * cw';
kernel = kernel ./ sum(kernel(:));
