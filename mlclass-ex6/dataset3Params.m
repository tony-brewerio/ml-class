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
% C and sigma above filled from calculation below
return;

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

% we test this values of C and sigma
C_variants = [0.03; 0.1; 0.3; 1; 3; 10; 30; 100];
sigma_variants = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% all possible C/sigma combinations
C_sigma = cartprod(C_variants, sigma_variants);
% store errros of C/sigma combinations here 
C_sigma_error = zeros(size(C_sigma, 1), 1);

for i = 1:size(C_sigma,1)
  % train on one C/sigma combination
  C = C_sigma(i, 1);
  sigma = C_sigma(i, 2);
  model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
  % store model's error
  predictions = svmPredict(model, Xval);
  C_sigma_error(i) = mean(double(predictions ~= yval));
end

% determine C/sigma combination with lowest error
[error, i] = min(C_sigma_error);
% use it as a result
C = C_sigma(i, 1);
sigma = C_sigma(i, 2);


% =========================================================================

end
