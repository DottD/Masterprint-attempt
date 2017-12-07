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
    'Dimension of a partial fingerprint'},...
    'Parameters',1,...
    {'15','0.75','[30,100]','128'});

%% Parameters and preallocations
N = str2double(user_pars{1});
m_stddev_thresh = str2double(user_pars{2});
freq_bounds = str2num(user_pars{3}); %#ok<ST2NM>
partial_dim = str2double(user_pars{4});
% Preallocate the lines span for selecting a partial fingerprint
partial_span = 0:partial_dim-1;
% Define a function to rescale to range [0,1]
rescaleStd = @(X) (X-min(X(:)))/(max(X(:))-min(X(:)));

%% Scan and process files
% Set up a filter to select only png images of the right thumb
file_list = rdir([nist_dir '/**/*_01.png']);
% Compute the total number of images
img_tot = numel(file_list);
img_done = 0;
% Create the progress dialog
progress_handle = waitbar(img_done/img_tot,['Computing... ',num2str(img_done),'/',num2str(img_tot)]);
for file = file_list'
    %% Load image in 8bit grayscale format
    img = double(fpimread(file.name));
    %% Crop out the whitespace
    [seg, mask] = borderRemove(img, m_stddev_thresh, freq_bounds);
    %% N images per fingerprint are selected
    % Prepare some matrices
    [mask_row, mask_col] = find(mask);
    mask_range_row = [min(mask_row(:)), max(mask_row(:))-partial_dim];
    tlc_row = zeros(N, 1);
    tlc_col = tlc_row;
    idx_to_draw = (1:N)';
    total_draws = 0;
    while (~isempty(idx_to_draw))
        %% Count how many total draws over the requested
        % This can be a useful information
        total_draws = total_draws + length(idx_to_draw);
        draws_ratio = total_draws/N;
        if draws_ratio >= 10
            break
        end
        %% Draw new rows
        new_tlc_row = randi(mask_range_row, length(idx_to_draw), 1);
        %% Draw new cols
        new_submask = mask(new_tlc_row, :);
        new_tlc_col = zeros(length(idx_to_draw), 1);
        for i = 1:length(idx_to_draw)
            [~, new_submask_col] = find(new_submask(i, :));
            new_range_col = [min(new_submask_col(:)), max(new_submask_col(:))-partial_dim];
            if new_range_col(1) >= new_range_col(2)
                new_tlc_col(i) = -1;
            else
                new_tlc_col(i) = randi(new_range_col);
            end
        end
        %% Gather with the previous
        tlc_row(idx_to_draw) = new_tlc_row;
        tlc_col(idx_to_draw) = new_tlc_col;
        %% Check if some row has too few elements
        too_few_cols = tlc_col < 0;
        if any(too_few_cols)
            % Redraw
            idx_to_draw = find(too_few_cols);
            continue;
        end
        %% Check repetitions
        [~, unique_row_idx, ~] = unique(tlc_row);
        unique_row_mask = false(size(tlc_row));
        unique_row_mask(unique_row_idx) = true;
        [~, unique_col_idx, ~] = unique(tlc_col);
        unique_col_mask = false(size(tlc_col));
        unique_col_mask(unique_col_idx) = true;
        ulc_repetitions = ~or(unique_row_mask, unique_col_mask);
        if any(ulc_repetitions)
            % Redraw
            idx_to_draw = find(ulc_repetitions);
            continue;
        end
        %% Check the ulcs to be inside the mask
        brc_row = tlc_row + partial_dim;
        brc_col = tlc_col + partial_dim;
        brc_ind = sub2ind(size(mask), brc_row, brc_col);
        brc_outside = ~mask(brc_ind);
        if any(brc_outside)
            % Redraw
            idx_to_draw = find(brc_outside);
            continue;
        end
        %% Empty the indices container
        idx_to_draw = [];
    end
    % Output information about the total number of draws
    disp(['Total number of draws: ', num2str(total_draws), ...
        ' draws performed over requested ratio: ', num2str(draws_ratio)]);
    %% Crop and save partial images
    for n = 1:N
        partial_img = img(tlc_row(n)+partial_span, tlc_col(n)+partial_span);
        % Save the image
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
    end
    %% Update the progress dialog
    img_done = img_done + 1;
    waitbar(img_done/img_tot, progress_handle, ['Computing... ',num2str(img_done),'/',num2str(img_tot)]);
end
elapsed = toc;
disp(['Elapsed ', num2str(elapsed), ' seconds']);
close(progress_handle);