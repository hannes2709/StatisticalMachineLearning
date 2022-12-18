library("MASS")
library("geometry")
source("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Project/algorithms.R")
#creating training data
n = 100
sigma = 1 
X1 = mvrnorm(n/2, mu = c(-3,0), Sigma = sigma * diag(2))
X2 = mvrnorm(n/2, mu = c(3,0), Sigma = sigma * diag(2))
Y1 = rep(-1, n/2)
Y2 = rep(1, n/2)
Y = c(Y1, Y2) #labels
Ycol = as.factor((Y + 1) / 2) #to plot data
plot(X[, 1], X[, 2], col = Ycol)
X = rbind(X1, X2) #classification data
w = c(1, 1) #starting point
eta = 0.01 #step size
iter = 10000 #number of iterations
a = gradient_descent(X, Y, w, eta, iter)
b = stochastic_gradient_descent(X, Y, w, eta, iter)
c = SVRG(X, Y, w, eta, iter, m = n/2)
d = SAGA(X, Y, w, eta, iter)
plot(unlist(b[1]), unlist(b[2]), type = "l", col = "red")
lines(unlist(c[1]), unlist(c[2]), col = "blue")
lines(unlist(a[1]), unlist(a[2]), col = "green")
lines(unlist(d[1]), unlist(d[2]), col = "brown")


plot(1:iter, unlist(b[2]), type = "l", col = "red")
lines(1:iter, unlist(c[2]), col = "blue")
lines(1:iter, unlist(a[2]), col = "green")
lines(1:iter, unlist(d[2]), col = "brown")
