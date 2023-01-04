library("MASS")
library("sigmoid")
library("geometry")
source("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Project/algorithms2.R")
#creating training data
p = 10
n = 1000
sigma = 5
X1 = mvrnorm(n/2, mu = c(-2, 0, 1, 2, 4, 0, 1, 2, 5, 1), Sigma = sigma * diag(p))
X2 = mvrnorm(n/2, mu = c(2, 5, -2, 1, 3, 0, 3, 3, 1, -1), Sigma = sigma * diag(p))
Y1 = rep(-1, n/2)
Y2 = rep(1, n/2)
Y = c(Y1, Y2) #labels
Ycol = as.factor((Y + 1) / 2) #to plot data
X = rbind(X1, X2) #classification data

####other data test### 
#library(MASS)

# Set the number of samples

# Set the design matrix
#x <- matrix(rnorm(n*2), n, 2)

# Set the true coefficients
#beta <- c(-1, 0.5)

# Generate the response variable
#Y <- rbinom(n, 1, 1/(1 + exp(-x %*% beta)))


############



plot(X[, 1], X[, 2], col = Ycol)
w = 1:p #starting point
eta = 0.1 #step size
iter = 100000 #number of iterations
m = n / 2 #hyperparameter in SVRG
a = gradient_descent(X, Y, w, eta, round(iter/n))
b = stochastic_gradient_descent(X, Y, w, eta, iter)
c = SVRG(X, Y, w, eta, round(iter / (m + n)), m)
d = SAGA(X, Y, w, eta, iter - n)
plot(unlist(b[1]), unlist(b[2]), type = "l", col = "red", log = "y")
lines(unlist(c[1]), unlist(c[2]), col = "blue")
lines(unlist(a[1]), unlist(a[2]), col = "green")
#lines(unlist(d[1]), unlist(d[2]), col = "brown")


#plot(1:iter, unlist(b[2]), type = "l", col = "red")
#lines(1:iter, unlist(c[2]), col = "blue")
#lines(1:iter, unlist(a[2]), col = "green")
#lines(1:iter, unlist(d[2]), col = "brown")
