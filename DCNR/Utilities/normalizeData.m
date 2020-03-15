function [descr] = normalizeData(Descr, normalization)
% normalizing the data    
% input: Descr-> train.descr or test.descr
%          normalization-> '255' or 'norm'
% output: descr -> the descr after normalize

    descr = zeros(size(Descr));
    switch normalization
        case '255'
            descr = Descr ./ 255.0;
        case 'norm'
%             for i = 1 : size(Descr, 2)
%                 descr(:, i) = Descr(:, i)./norm(Descr(:, i), 2);
%             end
            descr = normc(Descr);
        case 'no'
            descr = Descr;
    end
    
end
