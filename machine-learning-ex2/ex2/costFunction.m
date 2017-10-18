function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = sigmoid(X*theta);
sum1 = 0;
for i = 1:m,
    sum1 = sum1 + y(i) * log(h(i)) + (1-y(i))*log(1-h(i));
end
J = -1/m * (sum1);

t = size(theta);
for j = 1:t,
    grad(j) = 1/m * sum((h-y) .* X(:,j));
end






% =============================================================

end
