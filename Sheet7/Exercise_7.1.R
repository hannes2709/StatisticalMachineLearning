#b) generating data
n = 100
X = rnorm(n, 0, 2)
e = rnorm(n, 0, sqrt(0.25))
Y = X^2 * cos(X) + e

#c) Nadarya Watson estimator
#Kernel function
K = function(x, X, h) {
  K = exp(-((x - X)/h)^2)
  return(K)
}

nw = function(x, X, Y, h, K) {
  #creating output vector
  nw = rep(0, length(x))
  for (j in 1:length(x)) {    
    #calculate sum of all weights
    sum_of_all_weights = 0
    for (k in 1:length(Y)) {
      sum_of_all_weights = sum_of_all_weights + K(x[j], X[k], h)
    }
    #calculate weights
    weights = rep(0, length(Y))
    for (k in 1:length(Y)) {
      weights[k] = K(x[j], X[k], h) / sum_of_all_weights
    }
    #calculate fitted value
    nw[j] = t(weights) %*% Y
  }
  return(nw)
}
x = seq(from = -10, to = 10, by = 0.1)
h = 0.03 #bandwidth
fitted_values = nw(x, X, Y, h, K)
y_true = x^2 * cos(x)
plot(X,Y)
lines(x, fitted_values)

# f)
bandwidth_grid = seq(from = 0.001, to = 5, by = 0.01)
LOO_CV_error = rep(0, length(Y))
for (i in 1:length(bandwidth_grid)) {
  #creating S
  S = matrix(nrow = length(Y), ncol = length(Y))
  for (j in 1:length(Y)) {
    weights_sum = 0
    for (l in 1:length(Y)) {
      weights_sum = weights_sum + K(X[j], X[l], bandwidth_grid[i])
    }
    for (k in 1:length(Y)) {
      S[j][k] = K(X[j], X[k], bandwidth_grid[i]) / weights_sum
    }
  }
  diag_S = S[row(S) == col(S)]
  A = diag(length(Y)) - diag(diag_S)
  LOO_CV_error[i] = 1 / length(Y) * norm(solve(A) %*% (Y - S %*% Y))
}

