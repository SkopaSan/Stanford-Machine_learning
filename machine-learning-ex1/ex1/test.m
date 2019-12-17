predictions = zeros(m, 1)
for i = 1:m
  predictions(i) = (theta(1) + theta(2)*X(m+i) - y(i))^2
endfor

summa = sum(predictions)

Ji = 1/(2*m) * summa