clc;
clear;

% Add necessary paths
addpath('D:/Program Files/MATLAB/R2023b/toolbox/spm12/');

% Load data
file_path = 'Combined.h5';

% Read the 'Combined' dataset from the HDF5 file
data = h5read(file_path, '/Combined')';  

% Display the size of the multimodal data matrix
% disp(size(data));

%%%%%%%%%%%%%%%%%%%%%%% Initialize optimization algorithm parameters %%%%%%%%%%%%%%%%%%%%%%%
maxsteps = 500;     % Maximum iteration steps
num_mods = 4;       % Number of modalities
nochange = 0.0001;  % Stop optimization when the change is smaller than this value
verbose = 1;        % Display detailed information
ncomps = 18;        % Number of extracted components
lrate = 0.0001;     % Initial learning rate
ann_stp = 0.95;     % Annealing step size

%%%%%%%%%%%%%%%%%%%%%%% Call optimization function for multimodal fusion and optimization %%%%%%%%%%%%%%%%%%%%%%%
[weights, A, W, sources] = optimizeinfomax(data, ncomps, maxsteps, num_mods, nochange, lrate, ann_stp, verbose);

% Finish
disp('Multimodal data processing completed.');