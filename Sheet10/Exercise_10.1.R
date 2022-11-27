library(e1071)
library(mvtnorm)
library(fields)
inp = read.csv("/Users/hannesgubler/Documents/R_Repositories/StatisticalMachineLearning/Sheet10/simclass1_train.csv", sep = ";")
#making a data frame
X = inp[, -1]
Y = inp[, 1]
dat = data.frame(Y = as.factor(Y), X = X[, c(2,1)])
#Plotting data
par(mar = c(3.1 ,3.1 ,1.6 ,1.5), mgp = c (1.7 ,0.6 ,0),
      font.main =1, cex.main =0.8)
plot(X, col = Y +1, xlab = "X1" , ylab = "X2", asp =1)
#fitting svm with gaussian kernel using the tune function of the e1071 package
#gamma is the kernel parameter (h in the RBF kernel),ranges is the cost of violation of the margin
set.seed(1)
tune.out <- tune ( svm , Y ~ . , data = dat ,
                     kernel = "radial" ,
                     ranges = list ( cost = c (0.01 ,0.1 ,1 ,10 ,100) ,
                                     gamma = c (100)))
best.svm <- tune.out $ best.model #best model chosen by 10 fold cross validation
summary(tune.out)
best.svm
#plot the classification boundary
xgrid = seq(from = -6, to = 6, by = 0.1)
ygrid = seq(from = -6, to = 6, by = 0.1)
xygrid = expand.grid(xgrid, ygrid)
colnames(xygrid) = c("X.X1", "X.X2")
pgrid = as.numeric(predict(best.svm, newdata = xygrid)) - 1
imp = as.image(Z = pgrid, ind = xygrid, nx = 100, ny = 100)
image(imp, col = two.colors(start = "blue", end = "red", alpha = 0.3))
xlim = range(X[, 1]) + c(-1,1)
ylim = range(X[, 2]) + c(-1,1)
zlim = c(0,1)
contour(imp, levels = 0.5, add = TRUE, lty = "dashed")
points(X, col = Y+1)
