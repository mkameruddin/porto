train <- read.csv("G:data/porto seguro/train.csv")
names(train)
str(train)
summary(train)
table(train[,2])

# replacing -1 with NA in variables
summary(train[,c(4,6,7,23,24,25,26,28,30,32,35,36,38)])


for ( x in c(4,6,7,23,24,25,26,28,30,32,35,36,38))
 { w <- which(train[,x]==-1)
train[w,x]<-NA}
summary(train[,c(4,6,7,23,24,25,26,28,30,32,35,36,38)])
write.csv(train,"G:data/porto seguro/train_na.csv",row.names=F)

###removing colummns (2) and rows(1 lakh) which contain NAs

train_na <- read.csv("G:data/porto seguro/train_na.csv")
names(train_na)
train1 <- train_na[,-c(26,28)]
train2 <- na.omit(train1)
write.csv(train2,"G:data/porto seguro/train_no_na.csv",row.names=F)
