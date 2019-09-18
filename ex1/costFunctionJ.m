function J = costFunctionJ(X,y,theta)
  
  % X es la matriz que contiene los ejemplos
  % y son los valores finales
  m = size(X,1);              %  numero de ejemplos de datos
  pred = X*theta;             %  predicciones de los todos los m ejemplos
  sqrErrors=(pred-y).^2;      %  errores
  J = 1/(2*m)*sum(sqrErrors);
endfunction
