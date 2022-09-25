testdata = read.csv("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet1/Data for the pratical application exercises-20220924/simreg1_test.csv", sep = ";")
traindata = read.csv("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet1/Data for the pratical application exercises-20220924/simreg1_train.csv", sep= ";")
head(traindata)

# Linear regression using least squares estimator from EX2
X <- cbind(rep(1,200),traindata$X) #Setting up design matrix X with a column of ones
betaHat = solve(t(X) %*% X) %*% t(X) %*% traindata$Y
fittedValues = X %*% betaHat

#Creating new X values to fit them with betaHat
Xseq <- cbind(rep(1,101),seq(from=0,to=1,by=0.01))
fittedValuesSeq <- Xseq%*%solve(t(X)%*%X)%*%t(X)%*%traindata$Y
plot(traindata)
lines(seq(from=0, to=1, by=0.01), fittedValuesSeq)
#Calculating error for Linear Model
square_residuals = (traindata$Y - fittedValues)^2
error = 1/length(fittedValues) * sum(square_residuals)
#Linear regression with transformed X (order up to k0)
k0 = 10
X = matrix(rep(1,200))
for (k in 1:k0) {
  X = cbind(X, traindata$X^k)
}
Xseq = rep(1,101)
fittedValues = X%*%solve(t(X)%*%X)%*%t(X)%*%traindata$Y
for (k in 1:k0) {
  Xseq = cbind(Xseq, seq(from=0, to=1, by=0.01))
}
fittedValuesSeq <- Xseq%*%solve(t(X)%*%X)%*%t(X)%*%traindata$Y
plot(traindata)
lines(seq(from=0, to=1, by=0.01), fittedValuesSeq)

square_residuals = (traindata$Y - fittedValues)^2
error = 1/length(fittedValues) * sum(square_residuals)
