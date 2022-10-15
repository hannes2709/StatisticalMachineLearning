
#######################################################
# linear regression on simulated data
#######################################################
rm(list=ls())

# load the training and test datasets
train <- read.csv('~file path here',sep=';')
test <- read.csv('~file path here',sep=';')

par(mfrow=c(1,1))
plot(train)
dim(train) #200 observations, 2 variables

# linear regression
# k=1
X <- cbind(rep(1,200),train$X)
Yhat <- X%*%solve(t(X)%*%X)%*%t(X)%*%train$Y #least squares estimate dervied from Ex 1(a)
# predict on a sequence and plot the result
Xseq <- cbind(rep(1,101),seq(from=0,to=1,by=0.01))
Yhatseq <- Xseq%*%solve(t(X)%*%X)%*%t(X)%*%train$Y
plot(train); lines(seq(from=0,to=1,by=0.01),Yhatseq)

# higher order k=k0
k0 <- 8
X <- matrix(rep(1,200),ncol=1); for (k in 1:k0) X <- cbind(X,train$X^k)
Yhat <- X%*%solve(t(X)%*%X)%*%t(X)%*%train$Y
# predict on a sequence and plot the result
Xseq <- rep(1,101); for (k in 1:k0) Xseq <- cbind(Xseq,seq(from=0,to=1,by=0.01)^k)
Yhatseq <- Xseq%*%solve(t(X)%*%X)%*%t(X)%*%train$Y
plot(train); lines(seq(from=0,to=1,by=0.01),Yhatseq)

# compute squared error loss for some values of k
ErrTest <- c()
ErrTrain <- c()
kmax <- 10
for (k0 in c(1:kmax)){
  X <- rep(1,200); for (k in 1:k0) X <- cbind(X,train$X^k)
  Yhat <- X%*%solve(t(X)%*%X)%*%t(X)%*%train$Y
  ErrTrain[k0] <- mean((Yhat-train$Y)^2)
  Xtest <- rep(1,1000); for (k in 1:k0) Xtest <- cbind(Xtest,test$X^k)
  Yhattest <- Xtest%*%solve(t(X)%*%X)%*%t(X)%*%train$Y
  ErrTest[k0] <- mean((Yhattest-test$Y)^2)
}
plot(1:kmax,ErrTest,col='red',type='b',ylim=range(c(ErrTrain,ErrTest)), xlab = 'k', ylab = "Error", main =
       "Relationship between k and the error")
lines(1:kmax,ErrTrain,col='blue',type='b')
legend("topright", c("Training Error", "Test Error"), col= c('blue','red'), lty = 1)


# there is a problem with k0>10: R cannot invert t(X)%*%X. This is because some columns of X are almost linear combinations of the others. 

