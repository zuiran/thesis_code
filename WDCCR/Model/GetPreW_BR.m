function W = GetPreW_BR(train, test)
% support by nsc
    train_descr  = train.descr;
    train_label  = train.label;
    test_descr   = test.descr;
    test_label   = test.label;
    clear train test;
    class_num  = length(unique(test_label));
    errors = zeros(class_num, length(test_label));
    
    for j = 1 : class_num
        temp = test_descr - train_descr(:,train_label==j)*(((train_descr(:,train_label==j)'...
               *train_descr(:,train_label==j)))\ train_descr(:,train_label==j)' * test_descr);
%             + 0.01*eye(sum(train_label==j))

        errors(j,:) = sum(temp.*temp, 1);
    end
    W = errors;
        
%     W = W ./ sum(W);
    W = W ./ max(W);
end

