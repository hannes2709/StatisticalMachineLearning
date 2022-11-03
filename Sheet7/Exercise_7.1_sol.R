##########################
###### EXERCISE 7.2
##########################

# Implementation of the Nadaraya-Watson estimator
nw <- function(x, X, Y, h, K = dnorm) {
  
  # Arguments
  # x: evaluation points
  # X: vector (size n) with the predictors
  # Y: vector (size n) with the response variable
  # h: bandwidth
  # K: kernel
  
  Kx <- rbind(sapply(X, function(Xi) K((x - Xi) / h) / h))
  
  # Weights
  W <- Kx / rowSums(Kx) # Column recycling!
  
  # Means at x ("drop" to drop the matrix attributes)
  drop(W %*% Y)
  
}


make_S <- function(x, X, Y, h, K = dnorm){
  
  Kx <- rbind(sapply(X, function(Xi) K((x - Xi) / h) / h))
  
  # Weights
  W <- Kx / rowSums(Kx) # Column recycling!
  
  return(W)
  
}

# Generate some data to test the implementation
set.seed(1)
n <- 200
eps <- rnorm(n, sd = 0.5)
m <- function(x) x^2 * cos(x)
X <- rnorm(n, sd = 2)
Y <- m(X) + eps

#Y <- ( 1 / ( 1 + exp(-Y) ) )  

x_grid <- seq(-10, 10, l = 500)

# Bandwidth
h <- 0.5

# Plot data
plot(X, Y)
rug(X, side = 1); rug(Y, side = 2)
lines(x_grid, m(x_grid), col = 1)
lines(x_grid, nw(x = x_grid, X = X, Y = Y, h = h), col = 2)
legend("top", legend = c("True regression", "Nadaraya-Watson"),lwd = 2, col = 1:2)


#points(x=X[( 1 / ( 1 + exp(-Y) ) > 0.5 ) ], Y[( 1 / ( 1 + exp(-Y) ) > 0.5 ) ] , pch=19)





# f^{-i}
S_ii <- diag( make_S(x = X, X = X, Y = Y, h = h) )

LOO_y_Sii <- (Y-nw(x = X, X = X, Y = Y, h = h) )/ (1-S_ii)

pred_LOO_y <- numeric() 

for (i in 1:n){
  pred_LOO_y[i] <- nw(x = X[i], X = X[-i], Y = Y[-i], h = h)
}

plot((Y-pred_LOO_y))
points(LOO_y_Sii,pch=19)

# they correspond ... 

# now to choose h.. LOO with squared loss... 

h_s <- seq(0.15, 0.5, length.out=50)

LOO_y_Sii_vec <- numeric()

for (i in 1:length(h_s)){
  S_ii <- diag( make_S(x = X, X = X, Y = Y, h = h_s[i]) )
  LOO_y_Sii_vec[i] <- mean (  ( (Y-nw(x = X, X = X, Y = Y, h = h_s[i]) )/ (1-S_ii) )^2 )
}


plot(y=LOO_y_Sii_vec, x=h_s, type='o', pch=19, xlab='Bandwidths', ylab='LOO CV score')

chosen_h <- h_s[which.min(LOO_y_Sii_vec)] # this is the chosen h

points( y=LOO_y_Sii_vec[which.min(LOO_y_Sii_vec)], x= chosen_h , col='red', pch=19)

# PLOT IT.... 

plot(X, Y)
rug(X, side = 1); rug(Y, side = 2)
lines(x_grid, m(x_grid), col = 1)
lines(x_grid, nw(x = x_grid, X = X, Y = Y, h = chosen_h), col = 2)
legend("top", legend = c("True regression", "Nadaraya-Watson"),lwd = 2, col = 1:2)



##########################
###### EXERCISE 7.3
##########################



# Plot data
plot(X, Y)
rug(X, side = 1); rug(Y, side = 2)
lines(x_grid, m(x_grid), col = 1)

lines(x_grid, nw(x = x_grid, X = X, Y = Y, h = h), col = 2)
legend("top", legend = c("True regression", "Nadaraya-Watson"),lwd = 2, col = 1:2)
points(x=X[( 1 / ( 1 + exp(-Y) ) > 0.5 ) ], Y[( 1 / ( 1 + exp(-Y) ) > 0.5 ) ] , pch=19)


Y_bin_obs <-  (1 / ( 1 + exp(-Y) ) >0.5 )*1  



plot(Y_bin_obs, x=X)

points( y=(nw(x=X, X=X, Y=Y_bin_obs, h=0.2)>0.5)*1 , x=X,  col='red', pch=3)

actual=Y_bin_obs
predicted <- (nw(x=X, X=X, Y=Y_bin_obs, h=0.2)>0.5)*1
predicted_Y <- nw(x=X, X=X, Y=Y_bin_obs, h=0.2)


cm = as.matrix(table(Actual = actual, Predicted = predicted)) # create the confusion matrix
cm


n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes


accuracy = sum(diag) / n 

precision = diag / colsums 
recall = diag / rowsums 

data.frame(precision, recall)

calculate_roc <- function(df, cost_of_fp, cost_of_fn, n=100) {
  tpr <- function(df, threshold) {
    sum(df$Predicted >= threshold & df$Actual == 1) / sum(df$Actual == 1)
  }
  fpr <- function(df, threshold) {
    sum(df$Predicted >= threshold & df$Actual == 0) / sum(df$Actual == 0)
  }
  cost <- function(df, threshold, cost_of_fp, cost_of_fn) {
    sum(df$Predicted >= threshold & df$Actual == 0) * cost_of_fp + 
      sum(df$Predicted < threshold & df$Actual == 1) * cost_of_fn
  }
  roc <- data.frame(threshold = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tpr <- sapply(roc$threshold, function(th) tpr(df, th))
  roc$fpr <- sapply(roc$threshold, function(th) fpr(df, th))
  roc$cost <- sapply(roc$threshold, function(th) cost(df, th, cost_of_fp, cost_of_fn))
  return(roc)
}

library(ggplot2)
library(grid)


plot_roc <- function(roc, threshold, cost_of_fp, cost_of_fn) {
  library(gridExtra)
  
  norm_vec <- function(v) (v - min(v))/diff(range(v))
  
  idx_threshold = which.min(abs(roc$threshold-threshold))
  
  col_ramp <- colorRampPalette(c("green","orange","red","black"))(100)
  col_by_cost <- col_ramp[ceiling(norm_vec(roc$cost)*99)+1]
  p_roc <- ggplot(roc, aes(fpr,tpr)) + 
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    coord_fixed() +
    geom_line(aes(threshold,threshold), color=rgb(0,0,1,alpha=0.5)) +
    labs(title = sprintf("ROC")) + xlab("FPR") + ylab("TPR") +
    geom_hline(yintercept=roc[idx_threshold,"tpr"], alpha=0.5, linetype="dashed") +
    geom_vline(xintercept=roc[idx_threshold,"fpr"], alpha=0.5, linetype="dashed")
  
  p_cost <- ggplot(roc, aes(threshold, cost)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=4, alpha=0.5) +
    labs(title = sprintf("cost function")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")
  
  sub_title <- sprintf("threshold at %.2f - cost of FP = %d, cost of FN = %d", threshold, cost_of_fp, cost_of_fn)
  
  grid.arrange(p_roc, p_cost, ncol=2, sub=textGrob(sub_title, gp=gpar(cex=1), just="bottom"))
}


predictions.df <- data.frame( Actual = actual, Predicted = predicted_Y )


# equal fp and fn
cost_of_fp <- 1
cost_of_fn <- 1
threshold <- 0.5
roc <- calculate_roc(predictions.df, cost_of_fp, cost_of_fn, n = 100)
which.min(roc$cost)
plot_roc(roc, threshold=roc$threshold[which.min(roc$cost)], cost_of_fp, cost_of_fn)

# unequal fp and fn

cost_of_fp <- 4
cost_of_fn <- 1
threshold <- 0.5
roc <- calculate_roc(predictions.df, cost_of_fp, cost_of_fn, n = 100)
which.min(roc$cost)
plot_roc(roc, threshold=roc$threshold[which.min(roc$cost)], cost_of_fp, cost_of_fn)

# Just the ROC curve
plot(y=roc$tpr, x=roc$fpr, type='l')
abline(v=0.2, lty='dashed')


library(pROC)
auc(predictions.df$Actual, predictions.df$Predicted)




######
### BONUS 
######



# now also vary bandwidth.... ?


bands <- seq(0.002,0.2, length.out=20)

roc_mat <- array(dim=c(100,4,20))

cost_of_fp <- 4
cost_of_fn <- 1

for (i in 1:length(bands)){
  chosen_band <- bands[i]
  
  actual <- numeric()
  predicted_Y <- numeric()
  
  for (j in 1:n){
    actual[j]=Y_bin_obs[j]
    predicted_Y[j] <- nw(x=X[j], X=X[-j], Y=Y_bin_obs[-j], h=chosen_band)
  }
  
  predictions.df <- data.frame( Actual = actual, Predicted = predicted_Y )
  roc_mat[,,i] <- as.matrix( calculate_roc(predictions.df, cost_of_fp, cost_of_fn, n = 100) )
}


plot( y= apply( roc_mat[,4,], 2, min ), x=bands, ylab='Min. Cost', xlab='Bandwidth', type='o')

# what is the best bandwidth?
band_choose <- which.min( apply( roc_mat[,4,], 2, min ) )

actual=Y_bin_obs
predicted_Y <- nw(x=X, X=X, Y=Y_bin_obs, h=bands[band_choose])
predictions.df <- data.frame( Actual = actual, Predicted = predicted_Y )

roc <- calculate_roc(predictions.df, cost_of_fp, cost_of_fn, n = 100) 
# what is the best threshold now with this chosen bandwidth?
roc[ which.min( roc_mat[,4,band_choose] ) ,1]

plot_roc(roc, threshold=roc[ which.min( roc_mat[,4,band_choose] ) ,1], cost_of_fp, cost_of_fn)