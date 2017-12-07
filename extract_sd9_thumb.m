% Scan the input folder and save in the output folder only the images that
% are related to thumbs (according to the NIST SD09 file name conventions).

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
%% Scan and process files
% Set up a filter to select only png images of the right thumb
file_list = rdir([nist_dir '/**/*_01.png']);
% Compute the total number of images
img_tot = numel(file_list);
img_done = 0;
%% Create the progress dialog
progress_handle = waitbar(img_done/img_tot,['Copying... ',num2str(img_done),'/',num2str(img_tot)]);
for file = file_list'
    %% Split the file name in its components
    [path, name, ext] = fileparts(file.name);
    %% Follow up user's preference
    if keep_struct
        % Leave the same folder tree as in the input database
        path = strrep(path, nist_dir, save_dir);
    else
        % Take the save folder as the path for each image
        path = save_dir;
    end
    % Create folder if necessary
    if exist(path, 'dir')~= 7
        mkdir(path);
    end
    %% Copy the file
    copyfile(file.name, fullfile(path, [name, ext]));
    %% Update the progress dialog
    img_done = img_done + 1;
    if mod(img_done, 50)==0
        waitbar(img_done/img_tot, progress_handle, ['Computing... ',num2str(img_done),'/',num2str(img_tot)]);
    end
end
disp(['Finished copying from ', nist_dir]);
close(progress_handle);