function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% run the follow loop turn out C=1, sigma=0.1

%paras = [0.01;0.03;0.1;0.3;1;3;10;30;];
%bias = 3;
%for alpha=1:8,
%  for beta=1:8,
%    tmp_C = paras(alpha);
%    tmp_sigma = paras(beta);
%    model = svmTrain(X, y, tmp_C, @(x1, x2) gaussianKernel(x1, x2, tmp_sigma));
%    predictions = svmPredict(model, Xval);
%    tmp_bias = mean(double(predictions ~= yval))
%    if tmp_bias < bias,
%      C = tmp_C;
%      sigma = tmp_sigma;
%      bias = tmp_bias;
%    end
%  end
%end
    
    







% =========================================================================

end
