function M = GetPreM(train)
%
    train_descr  = train.descr;
    train_label  = train.label;
    clear train;
    
    M = [];  %% not good
    T_label = unique(train_label);
    for i = 1 : length(T_label)
        X_i = train_descr(:, train_label==T_label(i));
        M = blkdiag(M, X_i'*X_i);
    end
    
end

