function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
 
options = optimset('GradObj', 'on', 'MaxIter', 100);
for i=1:num_labels
	initial_theta = zeros(size(X, 2), 1);
	[theta] = fmincg(@(t)(lrCostFunction(t, X,y==i, lambda)), initial_theta, options);
	all_theta(i,:) = theta';
end

%Just a some practice
%{
X_0=X_1=X_2=X_3=X_4=X_5=X_6=X_7=X_8=X_9=[];
y_0=y_1=y_2=y_3=y_4=y_5=y_6=y_7=y_8=y_9=[];
for i = 1:m
  if y(i) == 10
    X_0 = [X_0;X(i,:)];
    y_0 = [y_0; 0];
  elseif y(i) == 1
    X_1 = [X_1;X(i,:)];
    y_1 = [y_1; 1];
  elseif y(i) == 2
    X_2 = [X_2;X(i,:)];
    y_2 = [y_2; 2];
  elseif y(i) == 3
    X_3 = [X_3;X(i,:)];
    y_3 = [y_3; 3];
  elseif y(i) == 4
    X_4 = [X_4;X(i,:)];
    y_4 = [y_4; 4];
  elseif y(i) == 5
    X_5 = [X_5;X(i,:)];
    y_5 = [y_5; 5];
  elseif y(i) == 6
    X_6 = [X_6;X(i,:)];
    y_6 = [y_6; 6];
  elseif y(i) == 7
    X_7 = [X_7;X(i,:)];
    y_7 = [y_7; 7];
  elseif y(i) == 8
    X_8 = [X_8;X(i,:)];
    y_8 = [y_8; 8];
  elseif y(i) == 9
    X_9 = [X_9;X(i,:)];
    y_9 = [y_9; 9];  
  endif
endfor

[J1 grad1] = lrCostFunction(all_theta(1,2:401)', X_0, y_0, 0.1);
%}




% =========================================================================


end
