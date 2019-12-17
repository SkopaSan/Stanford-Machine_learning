function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%Unvectorized implementation
%h = zeros(m, 1);
%for i = 1:m
%  h(i) = (theta(1) + theta(2)*X(m+i) - y(i))^2;
%endfor
%summa = sum(h);
%J = 1/(2*m) * summa;

%Vectorized implementation
h = X * theta; 
sqrErrors = (h-y).^2; 
J = 1/(2*m) * sum(sqrErrors);

% =========================================================================

end
