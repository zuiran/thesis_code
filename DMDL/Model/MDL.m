function varargout = MDL(params, train)
% mutli-dictionary learning for linear regression-based classification
% 2019-12-27
    train_descr  = train.descr;
    train_label   = train.label;
    clear train;
    
    num_Vi = params.num_Vi;
    num_G  = params.num_G;
%     lambda = params.lambda;
    maxIter = params.maxIter;
%     norm_A = 'l2'; % params.norm_A;
    uLabel = unique(train_label);
    num_class = length(uLabel);
    [dim] = size(train_descr, 1);
    
    %% initialize dictionary M, V, G
    M = zeros(dim, num_class);
    for i = 1 : num_class
        M(:, i) = mean(train_descr(:,train_label==uLabel(i)), 2);
        var = train_descr(:,train_label==uLabel(i)) - M(:, i);
        if num_Vi > size(var,2)
           var(:,end+1:num_Vi) = rand(dim, num_Vi-size(var,2));
        end
        V{i} = normc(var(:, 1:num_Vi));
    end
%     G = normc(rand(dim, num_G));
    G = zeros(dim, num_G);
%     M = zeros(size(M));
    %% update A and V, G alternatively
    [A, ~] = updateAandV(M, V, G, train_descr, train_label, uLabel, params);
    Loss = computeLoss(M, V, G, A, train_descr);
    for iter = 1 : maxIter
        [A, V] = updateAandV(M, V, G, train_descr, train_label, uLabel, params);
%         if iter == 1
%            Loss(iter) = computeLoss(M, V, G, A, train_descr);
%            fprintf('Iter 0: loss = %.4f.\n', loss);
%         end
%         V = V1;
        G = updateG(M, V, G, A, train_descr, num_G);
        Loss(iter+1) = computeLoss(M, V, G, A, train_descr);
%         fprintf('Iter %d: loss = %.4f.\n', iter,loss);
    end
%     loss = computeLoss(M, V, G, A, train_descr);
%     fprintf('Iter %d: loss = %.4g.\n', iter,loss);
    
    varargout{1} = M;
    varargout{2} = V;
    varargout{3} = G;
    varargout{4} = A;
    varargout{5} = Loss;
end

function [A, V] = updateAandV(M, V, G, train_descr, train_label, uLabel, params)
    
    A = cell(length(uLabel), 3);
    for i = 1 : length(uLabel)
        Xi = train_descr(:, train_label==uLabel(i));
        Dict = [M(:,i), V{i}, G];
        Alpha =  (Dict'*Dict + params.lambda*eye(size(Dict,2))) \ Dict'*Xi;
        A{i,1} = Alpha(1,:);
        A{i,2} = Alpha(2:params.num_Vi+1,:);
        A{i,3} = Alpha(params.num_Vi+2:end,:);
        Delta = Xi - M(:,i)*A{i,1} - G*A{i,3};
        for j = 1 : params.num_Vi
%             Delta_j = Delta - V{i}(:, 1:end ~= j)*A{i,2}(1:end ~= j, :);
            Delta_j = Delta - V{i}*A{i,2} + V{i}(:,j)*A{i,2}(j,:);
            V{i}(:, j) = normc(Delta_j*A{i,2}(j,:)');
        end
    end
end

function [G] = updateG(M, V, G, A, train_descr, num_G)
    
%     expand_Am = [];
%     expand_Av = [];
%     expand_V  = [];
%     for i = 1 : length(uLabel)
%         expand_Am = blkdiag(expand_Am, A{i}(1,:));
%         expand_Av = blkdiag(expand_Av, A{i}(2:params.num_Vi+1,:));
%         expand_V  = cell2mat(V);;
%     end
    expand_Am = blkdiag(A{:, 1});
    expand_Av = blkdiag(A{:, 2});
    expand_Ag = cell2mat(A(:,3)');
    expand_V  = cell2mat(V);
%     Delta = train_descr - M*expand_Am - expand_V*expand_Av;
    Delta = train_descr - M*expand_Am - expand_V*expand_Av;
    for j = 1 : num_G
%         Delta_j = Delta - G(:, 1:end ~= j)*expand_Ag(1:end ~= j, :);
        Delta_j = Delta - G*expand_Ag + G(:,j)*expand_Ag(j,:);
        G(:, j) = normc(Delta_j*expand_Ag(j,:)');
    end
end

function loss = computeLoss(M, V, G, A, train_descr)
    
    expand_Am = blkdiag(A{:, 1});
    expand_Av = blkdiag(A{:, 2});
    expand_Ag = cell2mat(A(:,3)');
    expand_V  = cell2mat(V);
    
    Dict = [M expand_V G];
    expand_A = [expand_Am; expand_Av; expand_Ag];
    
    Res = train_descr - Dict*expand_A;
    loss = sum(Res(:).^2) / size(train_descr,1);
end


