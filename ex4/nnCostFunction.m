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
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
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

% ------------------------------------------------------------------------------
% PARTE 1 (J without regularization)
  X = [ones(m,1),X]; %anyado 1 a la primera columna de X
  a1 = X;
  
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2,1),1), a2]; %anyado 1 a la primera columna + bios unit
  
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  
  salH_x = a3;
  
  % Dado que necesitamos un vector de 0's y 1's para una clasificacion
    %multiple : 
    vec_y = zeros(m,num_labels);
    for i = 1:m
      vec_y(i,y(i))=1;
    endfor
    %cabe destacar que vectorizado, el bucle podria reemplazarse por:
    % vec_y = (1:num_labels)==y;
    % Calculo J
    J = (1/m) * sum(sum((-vec_y.*log(salH_x))-((1-vec_y).*log(1-salH_x))));
% ------------------------------------------------------------------------------
% PARTE 2 (Backpropagation gradiente sin reguolarizacion)
  % Para una solucion con bucle debemos analizar capa por capa la red neuronal
  for t = 1:m
    %capa 1
    a1 = X(t,:)';
    %capa 2
    z2 = Theta1 * a1;
    a2 = [1;sigmoid(z2)]; % capa oculta `+ bios(1)
    %capa 3
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    %A diferencia que antes, opto por una implementacion vectorizada
    vecY = (1:num_labels)'==y(t);
    %Ahora calculo los valores de delta.
    %Cabe destacar que solo calculare delta3 y delta2 ya que no asociamos error
    % en los datos de entrada
    delta3 = a3 - vecY;
    
    delta2 = (Theta2' * delta3).*[1; sigmoidGradient(z2)];
    delta2 = delta2(2:end);
    
    %Actualizacion capital delta
    Theta1_grad= Theta1_grad + (delta2 * a1');
    Theta2_grad= Theta2_grad + (delta3 * a2');
  endfor
  Theta1_grad = (1/m) * Theta1_grad;
  Theta2_grad = (1/m) * Theta2_grad;
% ------------------------------------------------------------------------------  
% PARTE 3 (anyadir termino de regularizacion en J y capital theta)
  reg_term = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2))
            + sum(sum(Theta2(:,2:end).^2)));
  J = J + reg_term; % coste regularizado
  
  % A continuacion calculo los gradientes de la regularizacion
  Theta1_grad_reg = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
  Theta2_grad_reg = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
  
  % Ahora sumo los gradientes regularizados a theta
  Theta1_grad= Theta1_grad + Theta1_grad_reg;
  Theta2_grad= Theta2_grad + Theta2_grad_reg;
% ------------------------------------------------------------------------------  
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
