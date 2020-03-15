function acc = MDL_Classifier(lambda_, test, M, V, G)
    test_descr = test.descr;
    test_label = test.label;
    clear test;
    
    uLabel = unique(test_label);
    num_class = length(uLabel);
    
    for j = 1 : num_class
        Dict = [M(:,j), V{j}, G];
        A = (Dict'*Dict + lambda_*eye(size(Dict,2))) \ Dict'*test_descr; 
        Res = test_descr - Dict*A;
        Err(j, :) = sum(Res.^2, 1);
    end
    [~, ind] = min(Err, [], 1);
    pre_label = uLabel(ind);
    acc = sum(pre_label == test_label) / length(test_label) * 100;
end




