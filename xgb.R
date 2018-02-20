#This is a minimal framework for training xgboost in R using caret to do the cross-validation/grid tuning
# and using the normalized gini metric for scoring. The # of CV folds and size of the tuning grid
# are limited to remain under kaggle kernel limits. To improve the score up the nrounds and expand
# the tuning grid.

library(data.table)
library(caret)
library(xgboost)
library(verification)


#dtrain <- read.csv("train_na_replaced_same.csv")
#dtest <- read.csv("test_na_replaced_final.csv")

dtrain <- read.csv("train_na.csv")
dtest <- read.csv("test_na.csv")

#dtrain <-dtrain[,-32] #removing var with 104 factors
#dtest <- dtest[,-31]

#dtrain <- dtrain[,-c(26,28)]
#dtest <- dtest[,-c(25,27)]


# collect names of all categorical variables
cat_vars <- names(dtrain)[grepl('_cat$', names(dtrain))]

# turn categorical features into factors
for (x in 1:length(cat_vars))
{ dtrain[,cat_vars[x]] <- as.factor(dtrain[,cat_vars[x]])
dtest[,cat_vars[x]] <- as.factor(dtest[,cat_vars[x]]) }
str(dtrain)
str(dtest)


# one hot encode the factor levels
dtrain_all <-dtrain
library(ade4)
library(data.table)

for (f in cat_vars){
  dtrain_dummy = acm.disjonctif(dtrain_all[f])
  dtrain_all[f] = NULL
  dtrain_all = cbind(dtrain_all, dtrain_dummy)
}

dtest_all <-dtest
for (f in cat_vars){
  dtest_dummy = acm.disjonctif(dtest_all[f])
  dtest_all[f] = NULL
  dtest_all = cbind(dtest_all, dtest_dummy)
}

dtrain <-dtrain_all
dtest <- dtest_all



# create index for train/test split
#train_index <- sample(c(TRUE, FALSE), size = nrow(dtrain), replace = TRUE, prob = c(0.8, 0.2))


train.index <- createDataPartition(dtrain$target, p = 0.8, list = FALSE)


# perform x/y ,train/test split.
x_train <- dtrain[train.index, 3:ncol(dtrain)]
y_train <- as.factor(dtrain[train.index, 'target'])

x_test <- dtrain[-train.index, 3:ncol(dtrain)]
y_test <- as.factor(dtrain[-train.index, 'target'])

# Convert target factor levels to 0 = "No" and 1 = "Yes" to avoid this error when predicting class probs:
# https://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret
levels(y_train) <- c("No", "Yes")
levels(y_test) <- c("No", "Yes")

# normalized gini function taked from:
# https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

# create the normalized gini summary function to pass into caret
giniSummary <- function (data, lev = "Yes", model = NULL) {
  levels(data$obs) <- c('0', '1')
  out <- normalizedGini(as.numeric(levels(data$obs))[data$obs], data[, lev[2]])  
  names(out) <- "NormalizedGini"
  out
}

# create the training control object. Two-fold CV to keep the execution time under the kaggle
# limit. You can up this as your compute resources allow. 
trControl = trainControl(
  method = 'repeatedcv',
  number=2,
  repeats = 2,
  summaryFunction = giniSummary,
  classProbs = TRUE,
  verboseIter = TRUE,
  allowParallel = TRUE)

# create the tuning grid. Again keeping this small to avoid exceeding kernel memory limits.
# You can expand as your compute resources allow. 
tuneGridXGB <- expand.grid(
  nrounds=c(150),
  max_depth = c(5),
  eta = c(0.05),
  gamma = c(1),
  colsample_bytree = c(0.75),
  subsample = c(0.50),
  min_child_weight = c(0)
  )

#start <- Sys.time()

# train the xgboost learner
xgbmod <- train(
  x = x_train,
  y = y_train,
  method = 'xgbTree',
  metric = 'NormalizedGini',
  trControl = trControl,
  tuneGrid = tuneGridXGB)


#print(Sys.time() - start)

# make predictions
preds <- predict(xgbmod, newdata = x_test, type = "prob")
preds_final <- predict(xgbmod, newdata = dtest, type = "prob")
preds1 <- predict(xgbmod, newdata = x_test, type = "raw")
preds_final1 <- predict(xgbmod, newdata = dtest, type = "raw")
table(preds1)
table(y_test,preds1)
table(preds_final1)


# convert test target values back to numeric for gini and roc.plot functions
levels(y_test) <- c("0", "1")
y_test_raw <- as.numeric(levels(y_test))[y_test]

# Diagnostics
print(xgbmod$results)
print(xgbmod$resample)

# plot results (useful for larger tuning grids)
#plot(xgbmod)

# score the predictions against test data
normalizedGini(y_test_raw, preds$Yes)

# plot the ROC curve
#roc.plot(y_test_raw, preds$Yes, plot.thres = c(0.02, 0.03, 0.04, 0.05))

# prep the predictions for submissions
sub <- data.frame(id = as.integer(dtest[, 'id']), target = preds_final$Yes)
w<- which(sub[,2]>0.5)
summary(sub[,2])
sub[w,1]
# write to csv
write.csv(sub, 'xgb_submission_.csv', row.names = FALSE)

##############
