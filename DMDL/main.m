clear; clc;
addpath Model Utilities

%% data set setting 
dataset.name = 'GT_32x32';
dataset.tr_num = 8;
dataset.random = 0;
dataset.normalization = '255';
[train, test] = loadDataset(dataset);

%% Learning Dictionary from class-specific training data  
params.num_Vi = 8;
params.num_G = 40;
params.maxIter = 20;
params.lambda = 0.05; % lambda of MDL can be fixed to 0.05
[M, V, G, ~, Loss] = MDL(params, train);
fprintf('Multi-Dictionary learning finished on %s with loss of %f.\n',dataset.name, Loss(end))

%% Classification 
lambda_ = 1; % choose otimized lambda_ for classififcation
accuracy = MDL_Classifier(lambda_, test, M, V, G);
fprintf('Accuracy of MDL on %s is %.2f%%.\n',dataset.name, accuracy)



