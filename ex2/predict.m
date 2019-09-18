function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
% La logica de la prediccion es la siguiente:
  %Es necesario conocer la hipotesis para poder conocer los valores
  %discretos 0 y 1 del training set, para ello operaremos la function
  %sigmoid de X de dimension m*(n+1) por teta de dimension (n+1)*1
  hyp = sigmoid(X*theta);
  % A continuacion sabemos que para predecir y = 1 el valor de la hipotesis
  % debe ser mayor de 0.5, si no el valor que se ha predicho sera y = 0
  p = hyp>=0.5
% =========================================================================
end
