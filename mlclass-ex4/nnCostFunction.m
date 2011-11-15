function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.  % %   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% for cost function
X = [ones(m, 1) X];
z1 = Theta1*X';
z1 = 1 ./ (1+(e .^ -z1));
a2 = [ones(1, size(z1, 2)); z1];
z2 = Theta2*a2;
a3 = 1 ./ (1+(e .^ -z2));
h = a3';
for i=1:m,
    index = y(i);
    vector_y = zeros(num_labels, 1);
    vector_y(index) = 1;
    delta = -vector_y'*log(h(i,:)') - (1-vector_y)'*log(1-h(i,:)');
    J += delta;
end
J = 1.0/m * J;

% for regularization
reg1 = 0;
reg2 = 0;
for j=1:hidden_layer_size,
  sum1 = sum(Theta1(j, 2:size(Theta1,2)) .^ 2);
  reg1 += sum1;
end
for j=1:num_labels,
  sum2 = sum(Theta2(j, 2:size(Theta2,2)) .^ 2);
  reg2 += sum2;
end
J = J + lambda*(reg1 + reg2) / (2*m);

% back propagation
delta2 = zeros(size(Theta2,1), size(Theta2, 2));
delta1 = zeros(size(Theta1,1), size(Theta1, 2));
for i=1:m,
  a1 = X(i, :); %a1 is the input x
  z2 = Theta1 * a1'; %z2 is the hidden layer 1
  a2 = 1 ./ (1+(e .^ -z2)); % a2 = g(z2)
  a2 = [ones(1, size(a2, 2)); a2];
  z3 = Theta2 * a2; %z3 is the hidden layer 2 (output)
  a3 = 1 ./ (1+(e .^ -z3));  % a3 = g(z3)
  yi = 1:num_labels;
  yi = yi == y(i);
  error3 = a3 - yi';
  error2 = (Theta2(:,2:end)' * error3) .* sigmoidGradient(z2);
  %error2 = error2(2:end);
  delta2 = delta2 + error3*a2';
  delta1 = delta1 + error2*a1;
end
delta2 = delta2 / m;
delta1 = delta1 / m;
Theta1_grad(:,1) = delta1(:,1);
Theta1_grad(:,2:end) = Theta1(:, 2:end) * (lambda / m) + delta1(:,2:end);
Theta2_grad(:,1) = delta2(:,1);
Theta2_grad(:,2:end) = Theta2(:, 2:end) * (lambda / m) + delta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
