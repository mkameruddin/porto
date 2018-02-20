

#library(data.table)
#library(caret)
library(xgboost)
#library(verification)


#dtrain <- read.csv("train_na_replaced_same.csv")
#dtest <- read.csv("test_na_replaced_final.csv")

#dtrain <- read.csv("train_na.csv")
#dtest <- read.csv("test_na.csv")

#dtrain <-dtrain[,-34] #removing var with 104 factors
#dtest <- dtest[,-33]

dtrain <- train[,-c(26,28)]
dtest <- test[,-c(25,27)]


# collect names of all categorical variables
cat_vars <- names(dtrain)[grepl('_cat$', names(dtrain))]

# turn categorical features into factors
for (x in 1:length(cat_vars))
{ dtrain[,cat_vars[x]] <- as.factor(dtrain[,cat_vars[x]])
dtest[,cat_vars[x]] <- as.factor(dtest[,cat_vars[x]]) }
str(dtrain)
str(dtest)

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

i <- sample(nrow(train),0.7*nrow(train))

dtrain1 <- xgb.DMatrix(as.matrix(dtrain_all[i,-c(1,2)]), 
                       label=(dtrain_all[i,2]),missing=NA)

dtrain2 <- xgb.DMatrix(as.matrix(dtrain_all[-i,-c(1,2)]),label=dtrain_all[-i,2],missing=NA)

dtest1 <- xgb.DMatrix(as.matrix(dtest_all[,-1]),missing=NA)


#model <- xgboost(data = dtrain1,
#                 missing = NA,nrounds = 5, objective = "binary:logistic")

#cv.res <- xgb.cv(data = dtrain1,label =dtrain[,2], nfold = 5,
 #              nrounds = 2,nthreads=6, objective = "binary:logistic")


watchlist <- list(train=dtrain1, test=dtrain2)

bst <- xgb.train(data=dtrain1, max_depth=6, eta=0.02,nrounds=450,subsample=0.5,
silent=0, watchlist=watchlist, eval_metric = "auc",
early.stop.round = 10,objective = "binary:logistic")

pred_train2 <- predict(bst, dtrain2)
summary(pred_train2)

preds <- predict(bst, dtest1)
summary(preds)

d <- data.frame(dtest_all[,1],preds)
names(d) <- c("id","target")
head(d)

write.csv(d, file=gzfile("xgb_model6_2v.csv.gz"),row.names=F)


###############################




importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

xgb.dump(bst, with_stats = T)

library(DiagrammeR)
xgb.plot.tree(model = bst)

library(igraph)
xgb.plot.deepness(model = bst)

