function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X*theta);
sum1 = 0;
sum2 = 0;
for i = 1:m,
    sum1 = sum1 + y(i) * log(h(i)) + (1-y(i))*log(1-h(i)) ;
end
for k = 2:size(theta),
    sum2 = sum2 + theta(k)*theta(k);
end
J = -1/m * (sum1) + (lambda/(2*m)) * sum2;

t = size(theta);
    grad(1) = 1/m * sum((h-y) .* X(:,1));
for j = 2:t,
    grad(j) = 1/m * sum((h-y) .* X(:,j)) + (lambda/m)*theta(j);
end





% =============================================================

end
