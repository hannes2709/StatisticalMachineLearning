library("tidyverse")
#reading data
testA <- scan("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet6/Classification Data for Exercise 6-20221027/classificationA.test", list ( x1 =0 , x2 =0 , y =0))
trainA <- scan("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet6/Classification Data for Exercise 6-20221027/classificationA.train", list ( x1 =0 , x2 =0 , y =0))
testB <- scan("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet6/Classification Data for Exercise 6-20221027/classificationB.test", list ( x1 =0 , x2 =0 , y =0))
trainB <- scan("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet6/Classification Data for Exercise 6-20221027/classificationB.train", list ( x1 =0 , x2 =0 , y =0))
testC <- scan("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet6/Classification Data for Exercise 6-20221027/classificationC.test", list ( x1 =0 , x2 =0 , y =0))
trainC <- scan("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet6/Classification Data for Exercise 6-20221027/classificationC.train", list ( x1 =0 , x2 =0 , y =0))
testA = data.frame(x1 = testA$x1, x2 = testA$x2, y = testA$y)
trainA = data.frame(x1 = trainA$x1, x2 = trainA$x2, y = trainA$y)
testB = data.frame(x1 = testB$x1, x2 = testB$x2, y = testB$y)
trainB = data.frame(x1 = trainB$x1, x2 = trainB$x2, y = trainB$y)
testC = data.frame(x1 = testC$x1, x2 = testC$x2, y = testC$y)
trainC = data.frame(x1 = trainC$x1, x2 = trainC$x2, y = trainC$y)

#a)
#scatterplot of all training sets colored by label
A <- ggplot ( data = trainA , mapping = aes ( x = x1 , y = x2 ,
                                           color = as.factor (y))) + 
  geom_point ()
B <- ggplot ( data = trainB , mapping = aes ( x = x1 , y = x2 ,
                                              color = as.factor (y))) + 
  geom_point ()
C <- ggplot ( data = trainC , mapping = aes ( x = x1 , y = x2 ,
                                              color = as.factor (y))) + 
  geom_point ()
plot(A)
plot(B)
plot(C)

#b) LDA for each data set
#calculating the means
piA = sum(trainA$y) / length(trainA$y)
piB = sum(trainB$y) / length(trainB$y)
piC = sum(trainC$y) / length(trainC$y)
trainA_1label = trainA[trainA$y == 1,]
trainB_1label = trainB[trainB$y == 1,]
trainC_1label = trainC[trainC$y == 1,]
trainA_0label = trainA[trainA$y == 0,]
trainB_0label = trainB[trainB$y == 0,]
trainC_0label = trainC[trainC$y == 0,]
mu_A_1label = c(mean(trainA_1label$x1), mean(trainA_1label$x2))
mu_B_1label = c(mean(trainB_1label$x1), mean(trainB_1label$x2))
mu_C_1label = c(mean(trainC_1label$x1), mean(trainC_1label$x2))
mu_A_0label = c(mean(trainA_0label$x1), mean(trainA_0label$x2))
mu_B_0label = c(mean(trainB_0label$x1), mean(trainB_0label$x2))
mu_C_0label = c(mean(trainC_0label$x1), mean(trainC_0label$x2))
#creating covariance matrices (dont need to distinguish between 1 labe and 0 label
#in LDA)
XA = data.frame(x1 = trainA$x1, x2 = trainA$x2)
XB = data.frame(x1 = trainB$x1, x2 = trainB$x2)
XC = data.frame(x1 = trainC$x1, x2 = trainC$x2)
sigma_A = cov(XA)
sigma_B = cov(XB)
sigma_C = cov(XC)
#calculating w and b using formula from lecture 5 page 29
w_A = solve(sigma_A) %*% (mu_A_1label - mu_A_0label)
w_B = solve(sigma_B) %*% (mu_B_1label - mu_B_0label)
w_C = solve(sigma_C) %*% (mu_C_1label - mu_C_0label)
b_A = log((1-piA) / piA) + 0.5 * t(mu_A_0label) %*% solve(sigma_A) %*% mu_A_0label +
  0.5 * t(mu_A_1label) %*% solve(sigma_A) %*% mu_A_1label
b_B = log((1-piB) / piB) + 0.5 * t(mu_B_0label) %*% solve(sigma_B) %*% mu_B_0label +
  0.5 * t(mu_B_1label) %*% solve(sigma_B) %*% mu_B_1label
b_C = log((1-piC) / piC) + 0.5 * t(mu_C_0label) %*% solve(sigma_C) %*% mu_C_0label +
  0.5 * t(mu_C_1label) %*% solve(sigma_C) %*% mu_C_1label
#computing decision boundaries (wx +v = 0)  LDA Boundary Function : x2 = ( - x1 * w [1] + b ) / w [2]
LDAcurve_A <- function ( x ) {
  (-x * -w_A[1] - b_A) / w_A[2]
}
LDAcurve_B <- function ( x ) {
  (-x * -w_B[1] - b_B) / w_B[2]
}
LDAcurve_C <- function ( x ) {
  (-x * -w_C[1] - b_C) / w_C[2]
}
Graph_LDA_A = A + stat_function(fun = LDAcurve_A, color = "black")
Graph_LDA_B = B + stat_function(fun = LDAcurve_B, color = "black")
Graph_LDA_C = C + stat_function(fun = LDAcurve_C, color = "black")
plot(Graph_LDA_A)
plot(Graph_LDA_B)
plot(Graph_LDA_C)

#d)
# Logistic Regression
# A data
logres <- glm(y ~ x1 + x2 , data = trainA , family = binomial )
summary(logres)$coef
# Logistic Regression Coefficients
m1 <- summary(logres)$coef[[2 ,1]] # Coefficient of x1
m2 <- summary(logres)$coef[[3 ,1]] # Coefficient of x2
mc <- summary(logres)$coef[[1 ,1]] # Constant term
# Logistic Regression Boundary Function : f ( x ) = -( mc + m1 * x1 ) / m2
logcurve_A <- function ( x ) {
  -(mc + m1 * x) / m2
}
Graph_logres_A = A + stat_function(fun = logcurve_A, color = "black")
#B data
logres <- glm(y ~ x1 + x2 , data = trainB , family = binomial)
summary(logres)$coef
# Logistic Regression Coefficients
m1 <- summary(logres)$coef[[2 ,1]] # Coefficient of x1
m2 <- summary(logres)$coef[[3 ,1]] # Coefficient of x2
mc <- summary(logres)$coef[[1 ,1]] # Constant term
# Logistic Regression Boundary Function : f ( x ) = -( mc + m1 * x1 ) / m2
logcurve_B <- function ( x ) {
  -(mc + m1 * x) / m2
}
Graph_logres_B = B + stat_function(fun = logcurve_B, color = "black")
# C data
logres <- glm(y ~ x1 + x2 , data = trainC , family = binomial )
summary(logres)$coef
# Logistic Regression Coefficients
m1 <- summary(logres)$coef[[2 ,1]] # Coefficient of x1
m2 <- summary(logres)$coef[[3 ,1]] # Coefficient of x2
mc <- summary(logres)$coef[[1 ,1]] # Constant term
# Logistic Regression Boundary Function : f ( x ) = -( mc + m1 * x1 ) / m2
logcurve_C <- function ( x ) {
  -(mc + m1 * x) / m2
}
Graph_logres_C = C + stat_function(fun = logcurve_C, color = "black")
plot(Graph_logres_A)
plot(Graph_logres_B)
plot(Graph_logres_C)

#f)
#calculating errors

