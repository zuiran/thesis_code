clear; clc;
addpath Model Utilities

%% data set setting 
dataset.name = 'GT_32x32';
dataset.tr_num = 6;
dataset.random = 0;
dataset.normalization = '255';
% dataset.crop = 'c';
% dataset.cRate = 0.1;
[train, test] = loadDataset(dataset);

%% noiseless image classification by WDCCR-BD or WDCCR-BR  
% lambda = 100; % choose optimized paramters for each method freely
% beta = 0.1;
% gamma = 0.1;
M = GetPreM(train);
W_i1 = GetPreW_BD(train, test);
accuracy1 = DCCR(lambda, gamma, beta, W_i1, M, train, test);
fprintf('Accuracy of WDCCR-BD on %s is %.2f%%.\n',dataset.name, accuracy1)
W_i2 = GetPreW_BR(train, test);
accuracy2 = DCCR(lambda, gamma, beta, W_i2, M, train, test);
fprintf('Accuracy of WDCCR-BR on %s is %.2f%%.\n',dataset.name, accuracy2)

%% noisy image classification by R-WDCCR-BD or R-WDCCR-BR  
dataset.crop = 'c';
dataset.cRate = 0.3;
[train, test] = loadDataset(dataset);
M = GetPreM(train);
W_i1 = GetPreW_BD(train, test);
accuracy1 = R_DCCR(lambda, gamma, beta, W_i1, M, train, test);
fprintf('Accuracy of R-WDCCR-BD on %s is %.2f%%.\n',dataset.name, accuracy1)
W_i2 = GetPreW_BR(train, test);
accuracy2 = R_DCCR(lambda, gamma, beta, W_i2, M, train, test);
fprintf('Accuracy of R-WDCCR-BR on %s is %.2f%%.\n',dataset.name, accuracy2)




