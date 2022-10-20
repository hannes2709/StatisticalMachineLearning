#Solutions
library(mfp)
library(glmnet)

#Exercise 4.1 (a) Cleaning data.
data("bodyfat")
data = bodyfat[,-c(1,2,4)]
x = model.matrix(siri~., data)[, -1]
y = data$siri

grid = 10^seq(from = -3, to = -2, length = 100)

#Exercise 4.1 (b) Doing LASSO and ridge regression
ridge.mod = glmnet(x = x, y = y, alpha = 0, lambda = grid)
plot(ridge.mod, xvar = "lambda", main = "Ridge Regresion")

lasso.mod = glmnet(x = x, y = y, alpha = 1, lambda = grid)
plot(lasso.mod, xvar = "lambda", main = "LASSO")

#NOTE: In the following lines, we do ridge regression. To do LASSO, set alpha = 1 in the glmnet functions.
# Exercise 4.1 (c) Simple Cross-validation
set.seed(5)                # To control randomness
valdn = sample(1:252,100)  # Generating validation set
train = (-valdn)           

#grid = 10^seq(from = -2, to = 0, length = 100)

ridge.train = glmnet(x = x[train,], y = y[train], alpha = 0, lambda = grid)
ridge.pred = predict(object = ridge.train, newx = x[valdn,])
meansqerror = function(x) mean((x-y[valdn])^2)
error.cv = apply(X = ridge.pred, MARGIN = 2, FUN = meansqerror) # Calculating CV error for each lambda
plot(x = grid, y = error.cv, xlab = "Lambda", ylab = "Error", main = "Simple Validation Plot for Ridge Regression") 

lasso.train = glmnet(x = x[train,], y = y[train], alpha = 1, lambda = grid)
lasso.pred = predict(object = lasso.train, newx = x[valdn,])
meansqerror = function(x) mean((x-y[valdn])^2)
error.cv = apply(X = lasso.pred, MARGIN = 2, FUN = meansqerror) # Calculating CV error for each lambda
plot(x = grid, y = error.cv, xlab = "Lambda", ylab = "Error", main = "Simple Validation Plot for LASSO") 

# Exercise 4.1 (e) LOO Cross-validation

ridge.train = function(j) glmnet(x = x[-j,], y = y[-j], alpha = 0, lambda = grid)
sq.error = sapply(X = 1:252, FUN = function(j) (y[j] - predict(object = ridge.train(j), newx = x[j,]))^2) # Calculating CV error for each lambda and j
error.cv = apply(X = sq.error, MARGIN = 1, FUN = mean) # Calculating CV error for each lambda
plot(x = grid, y = error.cv, xlab = "Lambda", ylab = "CV Error", main = "LOO CV Plot for Ridge Regression")

lasso.train = function(j) glmnet(x = x[-j,], y = y[-j], alpha = 1, lambda = grid)
sq.error = sapply(X = 1:252, FUN = function(j) (y[j] - predict(object = lasso.train(j), newx = x[j,]))^2) # Calculating CV error for each lambda and j
error.cv = apply(X = sq.error, MARGIN = 1, FUN = mean) # Calculating CV error for each lambda
plot(x = grid, y = error.cv, xlab = "Lambda", ylab = "CV Error", main = "LOO CV Plot for LASSO")

# Exercise 4.1 (f) K-fold Cross-validation
# Partitioning data into K blocks
n = 252
K = 10
N = n %/% K 

#B is the list of blocks 
B = lapply(X = 0:(K-2), FUN = function(k) k*N + (1:N))
B[[10]] = ((K-1)*N+1):n 

ridge.train = function(b) glmnet(x = x[-b,], y = y[-b], alpha = 0, lambda = grid)
sq.error = sapply(B, function(b) apply((y[b] - predict(object = ridge.train(b), newx = x[b,]))^2, 2, mean)) # Calculating CV error for each lambda and block
error.cv = apply(sq.error, 1, mean) # Calculating CV error for each lambda
plot(x = grid, y = error.cv, xlab = "Lambda", ylab = "CV Error", main = "K-Fold CV Plot for Ridge Regression")

lasso.train = function(b) glmnet(x = x[-b,], y = y[-b], alpha = 1, lambda = grid)
sq.error = sapply(B, function(b) apply((y[b] - predict(object = lasso.train(b), newx = x[b,]))^2, 2, mean)) # Calculating CV error for each lambda and block
error.cv = apply(sq.error, 1, mean) # Calculating CV error for each lambda
plot(x = grid, y = error.cv, xlab = "Lambda", ylab = "CV Error", main = "K-Fold CV Plot for LASSO")

