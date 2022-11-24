# attach necessary packages
library(ISLR)           # data sets
library(rpart)          # decision tree methodology
library(rpart.plot)     # decision tree visualization
library(randomForest)   # random forest methodology

# read in Heart data set
Heart <- read.csv("http://faculty.marshall.usc.edu/gareth-james/ISL/Heart.csv")
# ensure reproducibility
set.seed(200)

Heart$AHD = factor(Heart$AHD) 


# transform salary variable to the logarithm of salary
Hitters$Salary <- log(Hitters$Salary)
# sample 70% of the row indices for training the models
trainHit <- sample(1:nrow(Hitters), 0.7*nrow(Hitters))
trainHeart <- sample(1:nrow(Heart), 0.7*nrow(Heart))


# fit decision tree to Hitters data
rpart(Hitters$Salary~Hitters$Years, data = Hitters)
# display results


# fit decision tree without controlling depth
# display results

# predict testing set - show first four predictions

# calculate SSE

# Take out the row identifier as a predictor

# fit classification decision tree

# display results

# predict the patients in the testing set




# fit bagging model

# get indices of complete cases

# fit bagging algorithm

# predict new values


# calculate sum of squared errors


# get indices of complete cases
# fit bagging algorithm

# predict new values


# fit bagging algorithm

# predict new values

# confusion matrix




# impute missing values instead of causing an issue

# fit random forest algorithm

# make predictions


# fit random forest algorithm
# make predictions

# confusion matrix




# plot OOB error as a function of number of trees

# list relative measure of variable importance

# plot variable importance

# partialplot


