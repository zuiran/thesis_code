function [accuracy, Alpha0, pre_label0] = DCCR(lambda, gamma, beta, W_i, M, train, test)
%    
    train_descr  = train.descr;
    train_label  = train.label;
    test_descr   = test.descr;
    test_label   = test.label;
    clear train test;
    
    errors = zeros(length(unique(train_label)), length(test_label));
%     soc_perClass = zeros(length(unique(test_label)), length(test_label));
    temp = train_descr'*train_descr;
    temp3 = train_descr'*test_descr;
    temp2 = (1+2*beta)*temp + (2*beta*(length(unique(train_label))-2)+gamma).*M;
%     tic
    for i = 1 : length(test_label)
        y = test_descr(:, i);
        W = [];
        ulabel = unique(train_label);
        for j = 1 : length(ulabel)
            l = ulabel(j);
%             W = [W ones(1,sum(train_label==l)).*W_i(j,i)]; 
            W = [W ones(1,sum(train_label==l)).*W_i(j,i)]; 

        end
%         W = diag(W);
        solution = (1+gamma)*(temp2 + lambda.*diag(W)) \ temp3(:, i);
        k = 1;
        for j = unique(train_label)  
            errors(k, i) = norm(y - train_descr(:,train_label==j)*solution(train_label==j));   
%             soc_perClass(k, i) = sum(solution(train_label==j));
            k = k + 1;
        end  
    end
%     toc
    [~, pre_label_index] = min(errors);
%     [~, pre_label_index] = max(soc_perClass);
    pre_label = pre_label_index;
    accuracy = sum(pre_label==test_label) / length(test_label) * 100;

    if nargout >1
        Alpha0 = Alpha;
    end

    if nargout > 2
        pre_label0 = [es_test_label; test_label];
    end

end
