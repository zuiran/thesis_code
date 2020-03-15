clear; clc;
addpath Model Utilities

%% data set setting 
dataset.name = 'ORL_32x32';
dataset.tr_num = 6;
dataset.random = 0;
dataset.normalization = '255';
% dataset.crop = 'c';
% dataset.cRate = 0.1;
[train, test] = loadDataset(dataset);

%% noiseless image classification by DCNR
lambda = 0.1; % choose optimized paramters for each method
beta = 0.1;
gamma = 0.1;
accuracy = DCNR(lambda, gamma, beta, train, test);
fprintf('Accuracy of DCNR on %s is %.2f%%.\n',dataset.name, accuracy)


%% noisy image classification by R-DCNR
dataset.crop = 'c';
dataset.cRate = 0.3;
[train, test] = loadDataset(dataset);
accuracy = R_DCNR(lambda, gamma, beta, train, test);
fprintf('Accuracy of R-DCNR on %s is %.2f%%.\n',dataset.name, accuracy)





