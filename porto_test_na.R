test<- read.csv("G:data/porto seguro/test.csv")

summary(test)

i<-c(4,6,7,23,24,25,26,28,30,32,35,36,38)
i-1
# replacing -1 with NA in variables
summary(test)


for ( x in i-1)
{ w <- which(test[,x]==-1)
test[w,x]<-NA}
summary(test[,i-1])

write.csv(test,"G:data/porto seguro/test_na.csv",row.names=F)
