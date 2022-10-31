#READ ME: Choose the testing and the training data sets below, 
#and then run the script to get the results.

#Libraries
library(ggplot2)
library(MASS)
library(emdbook)
library(rgl)
library(reshape2)
library(gtools)

#Training Data
train <- scan("classificationA.train", list(x1=0,x2=0,y=0))
train <- data.frame(x1=train$x1,x2=train$x2,y=train$y)

n = length(train$y)

#Testing Data
test <- scan("classificationA.train", list(x1=0,x2=0,y=0))
test <- data.frame(x1=test$x1,x2=test$x2,y=test$y)

n_test = length(test$y)

attach(train)

#Training Data Parameters
K = 2
n1 = sum(y)
n0 = n - n1

#Computation of MLE for QDA and LDA
#Computing MLE of Pi
pi = n1/n

#Computing the MLEs of the means
mu1 <- c((sum(y*x1))/n1,(sum(y*x2))/n1)
mu0 <- c((sum(x1 - y*x1))/n0, (sum(x2 - y*x2))/n0)

#Computing the MLE of Sigma1
Sigma1xx = (sum(x1*x1*y))/n1 - mu1[[1]]*mu1[[1]]
Sigma1xy = (sum(x1*x2*y))/n1 - mu1[[1]]*mu1[[2]]
Sigma1yx = Sigma1xy
Sigma1yy = (sum(x2*x2*y))/n1 - mu1[[2]]*mu1[[2]]

Sigma1c = c(Sigma1xx, Sigma1xy, Sigma1yx, Sigma1yy)

#Computing the MLE of Sigma0
Sigma0xx = (sum(x1*x1) - sum(x1*x1*y))/n0  - mu0[[1]]*mu0[[1]]
Sigma0xy = (sum(x1*x2) - sum(x1*x2*y))/n0  - mu0[[1]]*mu0[[2]]
Sigma0yx = Sigma0xy
Sigma0yy = (sum(x2*x2) - sum(x2*x2*y))/n0  - mu0[[2]]*mu0[[2]]

Sigma0c = c(Sigma0xx, Sigma0xy, Sigma0yx, Sigma0yy)

#The MLEs of Sigma1 and Sigma0 in QDA and Sigma in LDA
Sigma1 <- matrix(Sigma1c, nrow = 2)
Sigma0 <- matrix(Sigma0c, nrow = 2)
Sigma <- (n0*Sigma0 + n1*Sigma1)/n

#LDA Boundary Parameters
b <- log((1-pi)/pi) + (mu0 %*% solve(Sigma, mu0))/2 - (mu1 %*% solve(Sigma, mu1))/2
w <- solve(Sigma, mu1-mu0)

#LDA Boundary Function: x2 = (- x1*w[1] + b)/w[2]
LDAcurve <- function(x) {
  (- x*w[1] + b)/w[2]
}

#Logistic Regression
logres <- glm(y ~ x1 + x2, data = train, family = binomial)

#Logistic Regression Boundary Parameters
m1 <- summary(logres)$coef[[2,1]]
m2 <- summary(logres)$coef[[3,1]]
mc <- summary(logres)$coef[[1,1]]

#Logistic Regression Boundary Function: x2 =  -(mc + m1 * x1)/m2
logcurve <- function(x) {
  -(mc + m1 * x)/m2
}

#QDA Contour
cont_QDA <- curve3d(  sum((c(x,y)-mu0)%*%solve(Sigma0, (c(x,y)-mu0))) 
                      - sum((c(x,y)-mu1)%*%solve(Sigma1, (c(x,y)-mu1)))-log(n1/n0), 
                      from = c(-6,-6), to = c(6,6), n=c(100,100), 
                      sys3d="none")
dimnames(cont_QDA$z) <- list(cont_QDA$x,cont_QDA$y)
M_QDA <- reshape2::melt(cont_QDA$z)
detach()

#Base Plot
G <- ggplot(data = train, mapping = aes(x = x1, y = x2, color = as.factor(y)))+ geom_point()
#Plot with LDA boundary
Graph_LDA <- G + stat_function(fun = LDAcurve, color = "black") 
#Plot with QDA boundary
Graph_QDA <- G + geom_contour(data=M_QDA,aes(x=Var1,y=Var2,z=value),breaks=0,linejoin = "round",colour="black")
#Plot with Logistic boundary
Graph_Logistic <- G + stat_function(fun = logcurve, color = "black")

#Misclassification Error QDA
DeltaQDA1 <- function(x, y) {
  - log(det(Sigma1))/2 - sum((c(x,y)-mu1)%*%solve(Sigma1, (c(x,y)-mu1)))/2 + log(pi)
}
DeltaQDA0 <- function(x, y) {
  - log(det(Sigma0))/2 - sum((c(x,y)-mu0)%*%solve(Sigma0, (c(x,y)-mu0)))/2 + log(1-pi)
}

#Misclassification Error LDA
DeltaLDA1 <- function(x, y) {
  sum((c(x,y))%*%solve(Sigma, mu1)) - sum((mu1)%*%solve(Sigma, mu1))/2 + log(pi)
}
DeltaLDA0 <- function(x, y) {
  sum((c(x,y))%*%solve(Sigma, mu0)) - sum((mu0)%*%solve(Sigma, mu0))/2 + log(1-pi)
}

#Misclassification Error for LDA
J <- c(1:n_test)
c=0
for (j in J) {
  if((DeltaLDA1(test$x1[[j]],test$x2[[j]]) > DeltaLDA0(test$x1[[j]],test$x2[[j]])) != test$y[[j]])
    c <- c+1
}
MissClassErrorLDA <- c/n_test * 100

#Misclassification Error for QDA
J <- c(1:n_test)
c=0
for (j in J) {
  if((DeltaQDA1(test$x1[[j]],test$x2[[j]]) > DeltaQDA0(test$x1[[j]],test$x2[[j]])) != test$y[[j]])
    c <- c+1
}
MissClassErrorQDA <- c/n_test * 100

#Misclassification Error for Logistic Regression
PredLogres <- predict(logres, test, type="response")

J <- c(1:n_test)
c <- 0
for (j in J) {
  if(as.integer(round(PredLogres[[j]])) != test$y[[j]])
    c <- c + 1
}
MissClassErrorLogRes <- c/n_test * 100

#Printing the Results
MissClassErrorLDA
MissClassErrorQDA
MissClassErrorLogRes

#To print the graphs, simply enter the name of the graph.
#For example: Graph_LDA
