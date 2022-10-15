library("mfp")#for data
library("tidyverse")
library("glmnet")#for ridge and lasso regression
data(bodyfat)
head(bodyfat)
#remove columns 1,2 and 4 of the bodyfat data
bodyfat <- bodyfat %>% select(-case, -brozek, -density)
x <- model.matrix(siri ~ ., bodyfat)[, -1] #the [, -1] is there to remove the 
#fist column, which is a column of ones for the intercept (what we don't need when
#we use "glmnet" later)
y = bodyfat$siri
grid <- 10^(seq(9, -3, length = 100))
ridge_mod <- glmnet(x,y, alpha = 0, lambda = grid) #alpha = 0 for ridge,
#alpha = 1 for lasso regression
dim(coef(ridge_mod))#we can see that the dimension depends on the grid and #predictors
plot(ridge_mod, xvar = "lambda", label = TRUE)#plot of the coefficient values over the lambda grid
#We can see that the bigger lambda, the smaller the coefficients (as expected)

#Now doing the same for lasso regression, hence using alpha = 1
lasso_mod <- glmnet(x,y, alpha = 1, lambda = grid)
plot(lasso_mod, xvar <- "lambda", label = TRUE)#we can see that the coefficients
#go to 0 quicker than in the ridge regression, which is as expected.

#Now sample a training and validation set out of the whole training set
set.seed(5)
training <- sample(1:252,152)
x_test = x[-training, ]
y_test <- y[-training]
#performing ridge and lasso regression on the training set
ridge_train <- glmnet(x[training, ], y[training], alpha = 0, lambda = grid)
lasso_train <- glmnet(x[training, ], y[training], alpha = 1, lambda = grid)
#evaluating the test MSE for each lambda in ridge and lasso
ridge_mse = rep(0, length(grid))#0 vector to store all the MSE for each value of lambda
lasso_mse = rep(0,length(grid))#MSE here is the validation error with squared loss function
for (i in 1:length(grid)) {
  #the predict function gives the predictions of the ridge regression for a 
  #desired value of lambda (denoted by s here)
  ridge_pred <- predict(ridge_train, s = grid[i], newx = x_test)
  lasso_pred <- predict(lasso_train, s = grid[i], newx = x_test)
  ridge_mse[i] <- mean((ridge_pred - y_test)^2)
  lasso_mse[i] = mean((lasso_pred - y_test)^2)
}
plot(log(grid), ridge_mse)
plot(log(grid), lasso_mse)
best_lambda_ridge <- which.min(ridge_mse)
best_lambda_lasso <- which.min(lasso_mse)
print(grid[best_lambda_ridge])#best lambda of the grid for ridge
print(ridge_mse[best_lambda_ridge])#MSE of the best lambda in the grid for ridge
print(grid[best_lambda_lasso])#best lambda of the grid for lasso
print(lasso_mse[best_lambda_lasso])#MSE of the best lambda in the grid for lasso
#obviously these results will change with a different seed since then we are using
#different test and training sets.

#leave one out cross validation for ridge
loo_mse = matrix(0, length(grid), length(y))
for (i in 1:length(y)) {
  new_bodyfat <- bodyfat[-c(i), ] #removing the i-th row of the data set
  new_x <- model.matrix(siri ~ ., new_bodyfat)[, -1]
  new_y <- new_bodyfat$siri
  ridge_train <- glmnet(new_x, new_y, alpha = 0, lambda = grid)
  for (j in 1:length(grid)) {
    ridge_pred <- predict(ridge_train, s = grid[j], newx = x[i, ])
    loo_mse[j,i] <-mean((ridge_pred - y[i])^2)
  }
}
#for each value of lambda (in the grid), which are the rows of loo_mse we now 
#calculate the mean of all validation errors (loo_mse)
loo_mse_final <- rep(0, length(grid))
for (i in 1:length(grid)) {
  loo_mse_final[i] <- mean(loo_mse[i, ])
}
plot(log(grid), loo_mse_final)
best_lambda_loo <- which.min(loo_mse_final)
print(grid[best_lambda_loo])
#To calculate the decision function, we would take the best lambda now and
#make a new ridge regression over the whole data set with that value of lambda




#Solutions
library(mfp)
library(glmnet)

#Exercise 4.1 (a) Cleaning data.
data("bodyfat")
data = bodyfat[,-c(1,2,4)]
x = model.matrix(siri~., data)[, -1]
y = data$siri

grid = 10^seq(from = -3, to = 3, length = 100)

#Exercise 4.1 (b) Doing LASSO and ridge regression
ridge.mod = glmnet(x = x, y = y, alpha = 0, lambda = grid)
plot(ridge.mod, xvar = "lambda", main = "Ridge Regresion")

lasso.mod = glmnet(x = x, y = y, alpha = 1, lambda = grid)
plot(lasso.mod, xvar = "lambda", main = "LASSO")

#NOTE: In the following lines, we do ridge regression. To do LASSO, set alpha = 1 in the glmnet functions.
# Exercise 4.1 (c) Simple Cross-validation
set.seed(1)                # To control randomness
valdn = sample(1:252,100)  # Generating validation set
train = (-valdn)           

grid = 10^seq(from = -2, to = 0, length = 100)

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

