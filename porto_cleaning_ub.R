library(xgboost)
library(mice)
library(parallel)

train <- read.csv("train.csv")
test <-read.csv("test.csv")

# replacing -1 with NA in variables in train and test data
summary(train)
i <- c(4,6,7,23,24,25,26,28,30,32,35,36,38)
summary(train[,i])
for ( x in i)
{ w <- which(train[,x]==-1)
train[w,x]<-NA}
summary(train[,i])

summary(test)
for ( x in i-1)
{ w <- which(test[,x]==-1)
test[w,x]<-NA}
summary(test[,i-1])

# collect names of all categorical variables
cat_vars <- names(train)[grepl('_cat$', names(train))]

# turn categorical features into factors
for (x in 1:length(cat_vars))
{ train[,cat_vars[x]] <- as.factor(train[,cat_vars[x]])
  test[,cat_vars[x]] <- as.factor(test[,cat_vars[x]]) }
str(train)
str(test)

write.csv(train,"train_na.csv",row.names=F)
write.csv(test,"test_na.csv",row.names=F)

####### Replacing NA in train data

t1 <- subset(train, train[,2] ==1)
t2 <- subset(train, train[,2] ==0)

t1[is.na(t1[,4]),4] <- 1
summary(t1[,4])
t1[is.na(t1[,6]),6] <- 0
summary(t1[,6])
t1[is.na(t1[,7]),7] <- 0
summary(t1[,7])
t1[is.na(t1[,23]),23] <- 0.8653 #median of t1[,23]
summary(t1[,23])
t1[is.na(t1[,24]),24] <- 11
summary(t1[,24])
t1[is.na(t1[,30]),30] <- 1
summary(t1[,30])
t1[is.na(t1[,32]),32] <- 2
summary(t1[,32])
t1[is.na(t1[,38]),38] <- 0.3748 #median of t1[,38]
summary(t1[,38])

t11 <- t1[,-c(26,28)] # rm ccat_03 and cat_05
table(is.na(t11))


t2[is.na(t2[,4]),4] <- 1
summary(t2[,4])
t2[is.na(t2[,6]),6] <- 1
summary(t2[,6])
t2[is.na(t2[,7]),7] <- 0
summary(t2[,7])
t2[is.na(t2[,23]),23] <- 0.8653 #median of t1[,23]
summary(t2[,23])
t2[is.na(t2[,24]),24] <- 11
summary(t2[,24])
t2[is.na(t2[,25]),25] <- 1
summary(t2[,25])
t2[is.na(t2[,30]),30] <- 1
summary(t2[,30])
t2[is.na(t2[,32]),32] <- 2
summary(t2[,32])
t2[is.na(t2[,35]),35] <- 3 #median
summary(t2[,35])
t2[is.na(t2[,36]),36] <- 0.3742 #median
summary(t2[,36])
t2[is.na(t2[,38]),38] <- 0.37 #median
summary(t2[,38])

t22 <- t2[,-c(26,28)] # rm ccat_03 and cat_05
table(is.na(t22))

t_final <- rbind(t11,t22)


write.csv(t_final,"train_na_replaced_final.csv",row.names=F)

####### Replacing NA in test data

test[is.na(test[,3]),3] <- 1
summary(test[,3])
test[is.na(test[,5]),5] <- 0
test[is.na(test[,6]),6] <- 0
test[is.na(test[,22]),22] <- 0.80
test[is.na(test[,23]),23] <- 11
test[is.na(test[,24]),24] <- 1
test[is.na(test[,29]),29] <- 1
test[is.na(test[,31]),31] <- 2
test[is.na(test[,34]),34] <- 3
test[is.na(test[,37]),37] <- 0.37

test1 <- test[,-c(25,27)]
table(is.na(test1))

write.csv(test1,"test_na_replaced_final.csv")


########
train1 <- train[,-c(26,28)]
test1<-test[,-c(25,27)]

#w<-which(is.na(train1[,23]))
train2 <- na.omit(train1)
table(is.na(train2))

h<-sample(nrow(train2),0.8*nrow(train2))
train2[,2] <-as.factor(train2[,2])
dtrain <- xgb.DMatrix(data.matrix(train2[h,-c(1,2)]),label=log(train2[h,2]+1))
dtest <- xgb.DMatrix(data.matrix(train2[-h,-2]),label=train2[-h,2])
watchlist <- list(eval = dtest, train = dtrain)

############

## A simple xgb.train example:
param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2, 
              objective = "binary:logistic", eval_metric = "auc")
bst <- xgb.train(param, dtrain, nrounds = 2, watchlist)


bst <- xgboost(data = dtrain, label = log(train2[h,2]+1), 
               max_depth = 2, eta = 1, nthread = 2, nrounds = 2, 
               objective = "binary:logistic")
pred <- predict(bst, agaricus.test$data)

train2[,2] <-as.factor(train2[,2])
bstDense <- xgboost(data = as.matrix(train2[,-c(1,2)]), label = train2[,2], max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

dtrain <- xgb.DMatrix(data = train2[,-c(1,2)], label = train2[,2])
bstDMatrix <- xgboost(data = dtrain, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")


## A simple xgb.train example:
param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2, 
              objective = "binary:logistic", eval_metric = "auc")
bst <- xgb.train(param, dtrain, nrounds = 2, watchlist)


## An xgb.train example where custom objective and evaluation metric are used:
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

# These functions could be used by passing them either:
#  as 'objective' and 'eval_metric' parameters in the params list:
param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2, 
              objective = logregobj, eval_metric = evalerror)
bst <- xgb.train(param, dtrain, nrounds = 2, watchlist)

#  or through the ... arguments:
param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2)
bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                 objective = logregobj, eval_metric = evalerror)

#  or as dedicated 'obj' and 'feval' parameters of xgb.train:
bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                 obj = logregobj, feval = evalerror)


## An xgb.train example of using variable learning rates at each iteration:
param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2)
my_etas <- list(eta = c(0.5, 0.1))
bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                 callbacks = list(cb.reset.parameters(my_etas)))


## Explicit use of the cb.evaluation.log callback allows to run 
## xgb.train silently but still store the evaluation results:
bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                 verbose = 0, callbacks = list(cb.evaluation.log()))
print(bst$evaluation_log)

## An 'xgboost' interface example:
bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, 
               max_depth = 2, eta = 1, nthread = 2, nrounds = 2, 
               objective = "binary:logistic")
pred <- predict(bst, agaricus.test$data)
