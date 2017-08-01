function [Xi, yi] = randPick(X, y, i)
%% Pick i examples from the training set

m = size(X,1);
count = min([i m]);

rndIndx = randperm(count);
Xi = X(rndIndx(1:count), :);
yi = y(rndIndx(1:count), :);

end