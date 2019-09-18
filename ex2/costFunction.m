function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%Tengo que devolver J= escalar y grad= matriz de (n+1)*1
z = X * theta; % Doy el valor para calcular la funcion sigmoid de tamano m*1
%Calculo la hipotesis
hyp = sigmoid(z);
%Calculo el escalar
J = (1/m)*sum((-y.*log(hyp))-((1-y).*log(1-hyp)))
%Dado que no conozco las entradas para calcular uno a uno el gradiente
% grad(1 ... n+1) lo calculo de forma general segun la hyp de entrada
grad = (1/m)*(X'*(hyp-y))

% =============================================================

end
