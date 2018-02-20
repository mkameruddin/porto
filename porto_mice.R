train <- read.csv("train.csv")

summary(train[,c(4,6,7,23,24,25,26,28,30,32,35,36,38)])

for ( x in c(4,6,7,23,24,25,26,28,30,32,35,36,38))
{ w <- which(train[,x]==-1)
train[w,x]<-NA}
summary(train[,c(4,6,7,23,24,25,26,28,30,32,35,36,38)])

str(train)
names(train)

train[,4] <- as.factor(train[,4])
train[,6] <- as.factor(train[,6])
train[,7] <- as.factor(train[,7])
train[,24] <- as.factor(train[,24])
train[,25] <- as.factor(train[,25])
train[,26] <- as.factor(train[,26])
train[,27] <- as.factor(train[,27])
train[,28] <- as.factor(train[,28])
train[,29] <- as.factor(train[,29])
train[,30] <- as.factor(train[,31])
train[,31] <- as.factor(train[,31])
train[,32] <- as.factor(train[,32])
train[,33] <- as.factor(train[,33])
train[,34] <- as.factor(train[,34])
str(train)

library(mice)
library(parallel)

one <- function(.){
  mice(train[1:10000,-1])
}

m <- mclapply(1, FUN=one,mc.cores=7)
m1 <- complete(m[[1]])
summary(m1)
m2 <-data.frame(train[,c(1,2)],m1)
write.csv(m2,"train_mice1.csv",row.names=F)
