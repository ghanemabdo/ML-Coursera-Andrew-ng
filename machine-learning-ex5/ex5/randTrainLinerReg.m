function [error_train, error_val] = randTrainLinerReg(X, y, Xval, yval, lambda, training_examples_max)

iter = 50;

% You need to return these variables correctly.
error_train = zeros(training_examples_max, 1);
error_val = zeros(training_examples_max, 1);

for j = 1:training_examples_max

err_tr = 0.0;
err_val = 0.0;

for i = 1:iter

	[Xi, yi] = randPick(X, y, j);
	[Xvali, yvali] = randPick(Xval, yval, j);
	theta = trainLinearReg(Xi, yi, lambda);
	err_tr += linearRegCostFunction(Xi, yi, theta, 0);
	err_val += linearRegCostFunction(Xvali, yvali, theta, 0);

end

error_train(j) = err_tr / iter;
error_val(j) = err_val / iter;

end

end
