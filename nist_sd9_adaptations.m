% Apply to the NIST SD9 the same operations performed by Roy, Memon & Ross
% Images are preprocessed by cropping out the whitespace and then 
% downscaling to 256 ? 256 pixels. To get partial finger-print samples, 
% a random 128 ? 128 region is selected every time an image is selected.

%% Initial setting
% Clear all
clear;
clc;
close all;
tic;
% Extend the search path to all the folders in the current directory
addpath(genpath('.'));

%% User interaction
% Ask the user for the db folder
nist_dir = uigetdir;
% Ask the user for the save folder
save_dir = uigetdir;
% Ask the user how to build the output path
keep_struct = strcmp(questdlg(...
    'Do you want to keep the same folder structure as the input db?',...
    'Output path structure',...
    'No'), 'Yes');
% Ask the user the parameters
user_pars = inputdlg({'Number of partial images per fingerprint',...
    'Segmentation - min stddev percentile threshold',...
    'Segmentation - frequency bounds',...
    'Size to which rescale each fingerprint after segmentation',...
    'Dimension of a partial fingerprint'},...
    'Parameters',1,...
    {'9','0.97','[30,100]','256','128'});

%% Parameters and preallocations
N = str2double(user_pars{1});
m_stddev_thresh = str2double(user_pars{2});
freq_bounds = str2num(user_pars{3}); %#ok<ST2NM>
ds_dim = str2double(user_pars{4});
partial_dim = str2double(user_pars{5});

% Decide how many images take based on the user's suggestion
im_rows = floor(sqrt(N));
im_cols = ceil(sqrt(N));
if im_rows * im_cols ~= N
    warning(['Selecting ',num2str(im_rows*im_cols),' images per fingerprint instead...']);
end
% Check consistence of partial_dim and N
if any([im_rows,im_cols]+partial_dim > ds_dim)
    warning(['Image ', file.name,' too small -> skipped']);
end
% Compute the limits of all the possible upperleft corner (ulc) positions
ulc_minlim = 1;
ulc_maxlim = ds_dim-partial_dim;
% Compute the limits for each partial image upperleft corner
ulc_row_step = floor((ulc_maxlim-ulc_minlim)/im_rows);
ulc_col_step = floor((ulc_maxlim-ulc_minlim)/im_cols);
% Preallocate the lines span for selecting a partial fingerprint
partial_span = 0:partial_dim-1;
% Define a function to rescale to range [0,1]
rescaleStd = @(X) (X-min(X(:)))/(max(X(:))-min(X(:)));

%% Scan and process files
% Set up a filter to select only png images of the right thumb
file_list = rdir([nist_dir '/**/*_01.png']);
% Compute the total number of images
img_tot = numel(file_list)*N;
img_done = 0;
% Create the progress dialog
progress_handle = waitbar(img_done/img_tot,['Computing... ',num2str(img_done),'/',num2str(img_tot)]);
for file = file_list'
    %% Load image in 8bit grayscale format
    img = double(fpimread(file.name));
    %% Crop out the whitespace
    img = borderRemove(img, m_stddev_thresh, freq_bounds);
    %% Downscale to 256 x 256 pixels
    img = cropToSquare(img);
    if any(size(img) < 256)
        warning(['Image ', file.name, ' is oversampled']);
    end
    img = imresize(img, [256, 256]);
    %% N images per fingerprint are selected
    n = 1;
    for i = 1:im_rows
        for j = 1:im_cols
            %% Selection of partial image
            % Compute the local minimum and maximum positions for the ulc
            ulc_loc_min = [ulc_minlim+(i-1)*ulc_row_step, ulc_minlim+(j-1)*ulc_col_step];
            ulc_loc_max = ulc_loc_min + [ulc_row_step, ulc_col_step];
            % Randomly select an ulc in that range
            ulc = round(rand(1,2).*(ulc_loc_max-ulc_loc_min)+ulc_loc_min);
            % Select a partial image with that ulc
            partial_img = img(ulc(1)+partial_span, ulc(2)+partial_span);
            %% Save the image
            [path, name, ext] = fileparts(file.name);
            if keep_struct
                % Leave the same folder tree as in the input database
                path = strrep(path, nist_dir, save_dir);
            else
                % Take the save folder as the path for each image
                path = save_dir;
            end
            % Create folder if necessary
            if exist(path,'dir')~= 7
                mkdir(path);
            end
            % Append partial image counter at the end of the name
            imwrite(rescaleStd(partial_img), fullfile(path, [name,'_',num2str(n), ext]));
            % Increase the partial image counter
            n = n+1;
            % Update the progress dialog
            img_done = img_done + 1;
            waitbar(img_done/img_tot, progress_handle, ['Computing... ',num2str(img_done),'/',num2str(img_tot)]);
        end
    end
end
elapsed = toc;
disp(['Elapsed ', num2str(elapsed), ' seconds']);