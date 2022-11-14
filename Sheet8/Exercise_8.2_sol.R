library(splines)
library(MultiKink) #for the data
library(ggplot2)   #for the plots
set.seed(1974)     #fix the random generator seed 

# (b)
data("triceps")   #load the dataset triceps

# (c)
tri.age.plot <- ggplot(triceps, aes(x=age, y=triceps)) +
  geom_point(alpha=0.55, color="black") + 
  theme_minimal() 
tri.age.plot

# (d)
###Piecewise regression

pred1 <- predict(lm(triceps~age, 
                    data = triceps[triceps$age<5,]))
pred2 <- predict(lm(triceps~age, 
                    data = triceps[triceps$age >=5 & triceps$age<10,]))
pred3 <- predict(lm(triceps~age, 
                    data = triceps[triceps$age>=10 & triceps$age<20,]))
pred4 <- predict(lm(triceps~age, 
                    data = triceps[triceps$age>=20 & triceps$age<30,]))
pred5 <- predict(lm(triceps~age, 
                    data = triceps[triceps$age>=30 & triceps$age<40,]))
pred6 <- predict(lm(triceps~age, 
                    data = triceps[triceps$age>=40,]))

tri.age.plot + 
  geom_line(data=triceps[triceps$age<5,], 
            aes(y = pred1, x=age), size = 1, col="blue") +
  geom_line(data=triceps[triceps$age >=5 & triceps$age<10,], 
            aes(y = pred2, x=age), size = 1, col="blue") +
  geom_line(data=triceps[triceps$age>=10 & triceps$age<20,], 
            aes(y = pred3, x=age), size = 1, col="blue") +
  geom_line(data=triceps[triceps$age>=20 & triceps$age<30,], 
            aes(y = pred4, x=age), size = 1, col="blue") +
  geom_line(data=triceps[triceps$age>=30 & triceps$age<40,], 
            aes(y = pred5, x=age), size = 1, col="blue") +
  geom_line(data=triceps[triceps$age>=40,], 
            aes(y = pred6, x=age), size = 1, col="blue") 


# to make the segments connect, we use the basis functions $ I((age-k) * (age>=k)), where I is indicator function
pred7 <- predict(lm(triceps~ age + I((age-5)*(age>=5)) +
                      I((age-10)*(age >= 10)) +
                      I((age-20)*(age >= 20)) +
                      I((age-30)*(age >= 30)) +
                      I((age-40)*(age >= 40)),
                    data = triceps))

tri.age.plot +
  geom_line(data=triceps, 
            aes(y = pred7, x=age), size = 1, col="blue") 


# (e) quadratic piecewise. Similar to (d) but just adapt your basis functions
pred.quad <- predict(lm(triceps~ age + I(age^2) + 
                          I((age-5)*(age>=5)) + I((age-5)^2*(age>=5)) +
                          I((age-10)*(age >= 10)) + I((age-5)^2*(age>=10)) +
                          I((age-20)*(age >= 20)) + I((age-5)^2*(age>=20)) +
                          I((age-30)*(age >= 30)) + I((age-5)^2*(age>=30)) +
                          I((age-40)*(age >= 40)) + I((age-5)^2*(age>=40)),
                        data = triceps))

tri.age.plot +
  geom_line(data=triceps, 
            aes(y = pred.quad, x=age), size = 1, col="blue")



# (f)
#linear model with the natural cubic splines function 
cub.splines.bs <- lm(triceps ~ bs(age, knots = c(5,10,20,30,40)), 
                     data=triceps)
summary(cub.splines.bs)


cub.splines.ns <- lm(triceps ~ ns(age, knots = c(5,10,20,30,40)), 
                     data=triceps)

summary(cub.splines.ns)

# (g)

# Notice that are less regression parameters for the natural spline due to the linearity restriction. We can see this in the plot.

tri.age.plot +
  stat_smooth(method = "lm", 
              formula = y~bs(x,knots = c(5,10,20,30,40)), 
              lty = 1, col = "blue") + 
  stat_smooth(method = "lm", 
              formula = y~ns(x,knots = c(5,10,20,30,40)), 
              lty = 1, col = "red")  

#(h)
tri.age.plot +
  stat_smooth(method = "lm", 
              formula = y~ns(x,knots = c(5,10,20,30,40)), 
              lty = 1, col = "red") + 
  stat_smooth(method = "lm", 
              formula = y~ns(x,df=6), 
              lty = 1, col = "yellow")

#(i)
library(caret)
set.seed(1001)

#repeated CV for the MSE
trC.lm <- trainControl(method = "repeatedcv", 
                       number = 10, 
                       repeats = 10)

#function to fit a spline with x degrees of freedom
my.spline.f <- function(x) {
  #need to construct the model formula
  spline.formula <- as.formula(paste("triceps ~ ns(age, df=",x, ")" ))                                          
  pol.model <- train(spline.formula,                   
                     data = triceps,            
                     method = "lm",
                     trControl = trC.lm)
  
  RMSE.cv = pol.model$results[2]                #extracts the RMSE
}

#RMSE
t(sapply(2:20, my.spline.f))                    #Computes the RMSE for splines
#with df degrees 2 to 20
###########################################################################
#if you want to plot the curves,
#it is tricky to get ggplot to work 
#within a loop. This is a solutions:
col.ran <- sample(colours(), 20)                  #colours for the lines
my.plot<- tri.age.plot                            #scatterplot
for (i in 2:20){
  #builds the stat_smooth with df=i
  loop_input <-  paste("stat_smooth(method = \"lm\", 
                          formula = y~ns(x,df=",i,"), 
                          lty = 1, col =\"",col.ran[i],"\", 
                          se = FALSE)", sep="")
  
  #updates the scatter plot with 
  #the new spline
  my.plot <- my.plot + eval(parse(text=loop_input))    
}

my.plot


# (j)

#smooth spline with automatic number of knots chosen
#and penalisation chosen by leave-one-out CV (this is the
#option cv=T, otherwise generalized’ cross-validation is used)
sspline <- smooth.spline(triceps$age, 
                         triceps$triceps, 
                         cv=TRUE) 

plot(triceps$age, triceps$triceps)
lines(sspline, col="blue")

predict(sspline, x=c(10,30))


sspline <- smooth.spline(triceps$age, 
                         triceps$triceps, lambda=.0001) 
plot(triceps$age, triceps$triceps)
lines(sspline, col="blue")


