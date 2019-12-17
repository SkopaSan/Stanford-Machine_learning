function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    %Unvectorized implementation
    %grad = zeros(m, 1);
    %grad_0 = zeros(m, 1);
    %for i = 1:m
    %  grad(i) = (theta(1) + theta(2)*X(m+i) - y(i)) * X(m+i);
    %  grad_0(i) = (theta(1) + theta(2)*X(m+i) - y(i));
    %endfor
    %all_grad = sum(grad);
    %all_grad_0 = sum(grad_0);
    
    %theta(1) = theta(1) - (alpha * (1/m) * all_grad_0);
    %theta(2) = theta(2) - (alpha * (1/m) * all_grad);

    
    h = X * theta;
    err = h - y;
    grad = X' * err;
    new_theta = alpha * 1/m * grad;
    theta = theta - new_theta;

    fprintf('%f\n', theta);







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
