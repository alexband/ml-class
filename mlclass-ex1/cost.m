data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
% Initialize some useful values
m = length(y); % number of training examples
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%J = sum((X*theta-y) .^ 2)/(2*m)
% =========================================================================
(X*theta-y)' * X(:,1)

end
computeCost(X, y, theta)
