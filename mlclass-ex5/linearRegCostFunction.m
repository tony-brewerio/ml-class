function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% difference between prediction and expectance, used in all formulas
error = X * theta - y;

% special theta, where first element is replaced with 0, 
% so grad calculation can be completely vectorized, and J is simpler  
theta0 = theta;
theta0(1) = 0;

% cost calculation
J = sumsq(error) + lambda * sumsq(theta0);
J /= 2 * m;

% gradient calculation
grad = X' * error + lambda * theta0;
grad /= m;

% =========================================================================

grad = grad(:);

end
