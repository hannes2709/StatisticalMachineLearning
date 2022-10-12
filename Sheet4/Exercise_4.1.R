library("mfp")
library("tidyverse")
library("glmnet")
data(bodyfat)
head(bodyfat)
#remove columns 1,2 and 4 of the bodyfat data
bodyfat <- bodyfat %>% select(-case, -brozek, -density)
x <- model.matrix(siri ~ ., bodyfat)[, -1] #the [, -1] is there to remove the 
#fist column, which is a column of ones for the intercept (what we don't need when
#we use "glmnet" later)
y = bodyfat$siri
grid <- 10^(seq(10, -2, length = 100))
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
set.seed(1100)
training <- sample(1:252,152)
x_test = x[-training, ]
y_test <- y[-training]
#performing ridge and lasso regression on the training set
ridge_train <- glmnet(x[training, ], y[training], alpha = 0, lambda = grid)
lasso_train <- glmnet(x[training, ], y[training], alpha = 1, lambda = grid)
#evaluating the test MSE for each lambda in ridge and lasso
ridge_mse = rep(0, 100) #0 vector to store all the MSE for each value of lambda
lasso_mse = rep(0,100)
for (i in 1:length(grid)) {
  #the predict function gives the predictions of the ridge regression for a 
  #desired value of lambda (denoted by s here)
  ridge_pred <- predict(ridge_train, s = grid[i], newx = x_test)
  lasso_pred <- predict(lasso_train, s = grid[i], newx = x_test)
  ridge_mse[i] = mean((ridge_pred - y_test)^2)
  lasso_mse[i] = mean((lasso_pred - y_test)^2)
}
plot(log(grid), ridge_mse)
plot(log(grid), lasso_mse)
best_lambda_ridge = which.min(ridge_mse)
best_lambda_lasso = which.min(lasso_mse)
print(grid[best_lambda_ridge])#best lambda of the grid for ridge
print(ridge_mse[best_lambda_ridge])#MSE of the best lambda in the grid for ridge
print(grid[best_lambda_lasso])#best lambda of the grid for lasso
print(lasso_mse[best_lambda_lasso])#MSE of the best lambda in the grid for lasso
#obviously these results will change with a different seed since then we are using
#different test and training sets.
