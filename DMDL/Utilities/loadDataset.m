function [train, test] = loadDataset(dataset)
% dataset.{name, trainPerClass_num, normalization}
%
    %% get data.mat's folder
    filename_dataset = [dataset.name '.mat'];

    address_dataset = fullfile('Mat', filename_dataset);
    
    if ~isfield(dataset, 'random')
        dataset.random = 0;
    end
    if ~isfield(dataset, 'crop')
        dataset.crop = 'n';
        dataset.cRate = 0;
    end
    
    %% a.load dataset.mat, get Data{descr, label} b.normalization c.separate test&train{descr, label} randomly 
    if ~strcmp(filename_dataset, 'PIE_expression.mat')
        load(address_dataset);              
        
        if dataset.random
            [train, test] = getTrainAndTest_random(Data, dataset.tr_num);
        else
            [train, test] = getTrainAndTest(Data, dataset.tr_num);
        end
        
        test.descr = random_crop(test.descr, dataset.crop, dataset.cRate);
        [test.descr] = normalizeData(test.descr, dataset.normalization);
        [train.descr] = normalizeData(train.descr, dataset.normalization);
    else
        load Mat\PIE_expression_res50train_2048;
        train = Data;
        load Mat\PIE_expression_res50test_2048;
        test = Data;
    end
end