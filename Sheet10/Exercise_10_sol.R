# attach necessary packages
library(ISLR)           # data sets
library(rpart)          # decision tree methodology
library(rpart.plot)     # decision tree visualization
library(randomForest)   # random forest methodology

library(e1071)
library(mvtnorm)
library(fields)
###################################
#         Exercise 1
###################################
### Kernel logistic regression


rm(list=ls())

tab1 <- read.csv('simclass1_train.csv',sep=';')

X <- tab1[,-1]
Y <- tab1[,1]

# randomize obs (needed for good cross validation later)
set.seed(1)
S <- sample(1:100,100)
X <- X[S,]
Y <- Y[S]

par(mar=c(3.1,3.1,1.6,1.5),mgp=c(1.7,0.6,0),font.main=1,cex.main=0.8)
plot(X,col=Y+1,xlab='X1',ylab='X2',asp=1)

## kernel SVM
dat <- data.frame(Y=as.factor(Y),X=X[,c(2,1)])
tune.out2 <- tune(svm,Y~., data=dat, kernel='radial',ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100),gamma=c(0.1,0.5,1,1.5,2,2.5,3)))
best.svm2 <- tune.out2$best.model
###my plot for svm
xgrid <- seq(from=-6,to=6,by=0.1)
ygrid <- seq(from=-6,to=6,by=0.1)
xygrid <- expand.grid(xgrid,ygrid)
colnames(xygrid) <- c('X.X1','X.X2')
pgrid <- as.numeric(predict(best.svm2,newdata=xygrid))-1
imp <- as.image(Z=pgrid,ind=xygrid,nx=100,ny=100)
image(imp,col = two.colors(start='blue',end='red',alpha=0.3),xlim=range(X[,1])+c(-1,1),ylim=range(X[,2])+c(-1,1),zlim=c(0,1))
contour(imp,levels=0.5,add=TRUE,lty='dashed')
points(X,col=Y+1)

#### KLR with Gaussian kernel (note that Y must contain values -1/+1)
Y2 <- 2*Y-1

# main function
fitKLR <- function(Y2,X,lambda,nu){ #need to estimate the coefficients a_i and the intercept alpha0. nu is the coeff of the Gaussian kernel
  K <- exp(-nu*rdist(X)^2)
  n <- length(Y2)
  a <- rep(0,n)
  err <- 100
  iter <- 0
  K_tilde <- cbind(rep(1,n),K)
  K0 <- rbind(rep(0,n+1),cbind(rep(0,n),K))
  a_tilde <- c(0,a) #contains the intercept (initial value at 0 here)
  while (err>10^-10 & iter<1000){
    p <- 1/(1+exp(-Y2*K_tilde%*%a_tilde)); p[p==0] <- 0.0001; p[p==1] <- 0.9999
    W <- diag(x=c(p*(1-p)))
    an <- solve(1/n*t(K_tilde)%*%W%*%K_tilde+2*lambda*K0)%*%t(K_tilde)%*%W%*%(K_tilde%*%a_tilde+diag(x=c(1/(p*(1-p))))%*%(Y2*(1-p)))/n
    err <- sum((a_tilde-an)^2)
    a_tilde <- an
    iter <- iter + 1 
  }
  return(a_tilde)
}

a0 <- fitKLR(Y2,X,0.01,1)
#note that the intercept is positive (0.018) so that we will predict 1 when far from the x_i

# Cross validation to find (lambda,nu) ### My function fitKLR doesn't work well when nu or lambda are too small
lambdaseq <- c(0.01,0.1,1,5,100)/100
nuseq <- c(0.5,0.8,1,1.5,2,5,10)
Err <- matrix(ncol=length(nuseq),nrow=length(lambdaseq))

D <- rdist(X) #rdist function is much faster than using matrix(dist(X))
inu <- ilambda <- 0
for (nu0 in nuseq){
  inu <- inu + 1
  ilambda <- 0
  for (lambda0 in lambdaseq){
    ilambda <- ilambda + 1
    errj <- c()
    for (j in 1:10){ #10 fold CV
      ii <- c( (10*(j-1)+1):(10*j) )
      Xcv <- X[-ii,]
      Y2cv <- Y2[-ii]
      acv <- fitKLR(Y2cv,Xcv,lambda0,nu0)
      #pred at other xs
      Kcv <- exp(-nu0*as.matrix(D[ii,-c(ii)])^2)
      predcv <- as.numeric((acv[1]+Kcv%*%acv[-1]) > 0)*2-1
      errj[j] <- mean(!(predcv==Y2[ii]))
    }		
    Err[ilambda,inu] <- mean(errj)
  }
}
rownames(Err) <- lambdaseq
colnames(Err) <- nuseq
Err

nuhat <- 1
lambdahat <- 0.01

a <- fitKLR(Y2,X,lambdahat,nuhat)
xgrid <- seq(from=-5,to=6,by=0.1)
ygrid <- seq(from=-4,to=7,by=0.1)
xygrid <- expand.grid(xgrid,ygrid)
K0 <- exp(-nuhat*rdist(as.matrix(xygrid),as.matrix(X))^2)
f0 <- a[1]+K0%*%a[-1]
p0 <- exp(f0)/(1+exp(f0))

imp0 <- as.image(Z=p0,ind=xygrid,nx=100,ny=100)
image(imp0,col = two.colors(start='blue',end='red',alpha=1),xlim=range(X[,1])+c(-1,1),ylim=range(X[,2])+c(-1,1),zlim=c(0,1))
contour(imp0,levels=c(.5),add=TRUE,lwd=2) #the decision boundary
contour(imp0,levels=seq(from=0.1,to=0.9,by=0.1),add=TRUE,lty='dashed') #plot some probability contours
points(X,col=Y+1)





###################################
#         Exercise 2
###################################
# read in Heart data set
Heart <- read.csv("Heart.csv")
# ensure reproducibility
set.seed(200)

Heart$AHD = factor(Heart$AHD) 


# transform salary variable to the logarithm of salary
Hitters$Salary <- log(Hitters$Salary)
# sample 70% of the row indices for training the models
trainHit <- sample(1:nrow(Hitters), 0.7*nrow(Hitters))
trainHeart <- sample(1:nrow(Heart), 0.7*nrow(Heart))


# fit decision tree to Hitters data
salfit <- rpart(formula = Salary ~ Years + Hits, data = Hitters[trainHit,], 
                method = "anova", control = rpart.control(maxdepth = 2))
# display results
prp(salfit)


# fit decision tree without controlling depth
salfit <- rpart(formula = Salary ~ Years + Hits, data = Hitters[trainHit,], method = "anova")
# display results
rpart.plot(salfit)

# predict testing set - show first four predictions
predict(salfit, Hitters[-trainHit,])[1:4]

# calculate SSE
(treeSSE <- sum((predict(salfit, Hitters[-trainHit,]) - Hitters$Salary[-trainHit])^2, na.rm = TRUE))

# Take out the row identifier as a predictor
Heart <- Heart[,-which(colnames(Heart) == "X")]
# fit classification decision tree
AHDfit <- rpart(AHD ~ ., data = Heart[trainHeart,], method = "class")
# display results
rpart.plot(AHDfit)

# predict the patients in the testing set
predict(AHDfit, Heart[-trainHeart,])[1:4]


(treeT <- table(predict(AHDfit, Heart[-trainHeart,], type = "class"),Heart$AHD[-trainHeart]))


# fit bagging model
# get indices of complete cases
fullHit <- (1:nrow(Hitters))[complete.cases(Hitters)]
# fit bagging algorithm
bag_salary <- randomForest(Salary ~ ., data = Hitters[intersect(fullHit,trainHit),],
                           mtry = (ncol(Hitters)-1), ntree = 5000)
# predict new values
predict(bag_salary, Hitters[-trainHit,])[5:8]


# calculate sum of squared errors
(bagSSE <- sum((predict(bag_salary, Hitters[-trainHit,]) - Hitters$Salary[-trainHit])^2, na.rm = TRUE))


# get indices of complete cases
fullHeart <- (1:nrow(Heart))[complete.cases(Heart)]
# fit bagging algorithm
bag_heart <- randomForest(AHD ~ ., data = Heart[intersect(fullHeart,trainHeart),], 
                          mtry = (ncol(Heart)-1), ntree = 5000)
# predict new values
predict(bag_heart,Heart[-trainHeart,])[5:8]# get indices of complete cases


fullHeart <- (1:nrow(Heart))[complete.cases(Heart)]
# fit bagging algorithm
bag_heart <- randomForest(AHD ~ ., data = Heart[intersect(fullHeart,trainHeart),], 
                          mtry = (ncol(Heart)-1), ntree = 5000)
# predict new values
predict(bag_heart,Heart[-trainHeart,])[5:8]

# confusion matrix
(bagT <- table(predict(bag_heart, Heart[-trainHeart,], type = "class"),Heart$AHD[-trainHeart]))




# impute missing values instead of causing an issue
rf_salary <- randomForest(Salary ~ ., data = Hitters[trainHit,], na.action = na.roughfix)

# fit random forest algorithm
rf_salary <- randomForest(Salary ~ ., data = Hitters[intersect(fullHit, trainHit),], 
                          ntree = 5000, importance = TRUE)
# make predictions
predict(rf_salary, Hitters[-trainHit,])[9:12]

(rfSSE <- sum((predict(rf_salary, Hitters[-trainHit,]) - Hitters$Salary[-trainHit])^2, na.rm = TRUE))


# fit random forest algorithm
rf_heart <- randomForest(AHD ~ ., data = Heart[intersect(fullHeart, trainHeart),], ntree = 5000)
# make predictions
predict(rf_heart, Heart[-trainHeart,])[9:12]

# confusion matrix
(rfT <- table(predict(rf_heart, Heart[-trainHeart,], type = "class"), Heart$AHD[-trainHeart]))




# plot OOB error as a function of number of trees
plot(rf_salary)

# list relative measure of variable importance
importance(rf_salary)

# plot variable importance
varImpPlot(rf_salary)

# partialplot
partialPlot(rf_salary, pred.data=Hitters[trainHit,], x.var='Hits')



