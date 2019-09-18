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
    
    % debo definir la variable x y la hipotesis h que conlleva 
    x = X(:,2)   %Tomo todos los elementos de la 2 columna de X(valor de x)
    % creo la hipotesis
    h = theta(1)+(theta(2)*x) % vectores 1-indexados
    
    % Calculo segun la formula theta0 y theta1
    theta_cero = theta(1)-alpha * (1/m) * sum(h-y)
    theta_uno = theta(2)-alpha*(1/m) * sum((h-y).*x)
    % Guardo el resultado en la variable theta
    theta = [theta_cero;theta_uno]

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
%  para ver el resultado, dibujo la grafica resultante
  disp(min(J_history));
end
