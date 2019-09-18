function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

  % Inicio las listas correspondientes a C y sigma
  list_C = [0.01 0.03 0.1 0.3 1 3 10 30]';
  list_sigma = [0.01 0.03 0.1 0.3 1 3 10 30]';

  tam_sigma = length(list_sigma);
  tam_c = length(list_C);
  % A continuacion creo la lista de prediccion de error
  prediccion_Error = zeros(tam_c,tam_sigma);
  % Recorro la lista de entrada C
  
  for i = 1:tam_c
    for j = 1:tam_sigma
      prueba_C = list_C(i);
      prueba_sigma = list_sigma(j);
      model = svmTrain(X,y,prueba_C,@(x1,x2)gaussianKernel(x1,x2,prueba_sigma));
      predictions= svmPredict(model,Xval);
      prediccion_Error(i,j)= mean(double(predictions~= yval));
    endfor
  endfor
  
  % Buscar fila y columna correspondiente al error minimo
  [values, row_index]= min(prediccion_Error);
  [~,col] = min(values);
  row = row_index(col);
  
  % Devuelvo C y sigma correspondientes al minimo
  C = list_C(row);
  sigma = list_sigma(col);
% =========================================================================

end
