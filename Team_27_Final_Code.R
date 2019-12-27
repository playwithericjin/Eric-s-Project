## 4740 Final Project Code
## Creator: Jia Zhang, Shaoqing Jin, Jiankun Chen

## Import data and attach package

data <- read.csv("C:/Cornell/4740/final/divorce.csv",header = TRUE)
library(glmnet)
library(MASS)
library(tree)
library(randomForest)
library(gbm)
library(caret)
library(mlbench)
library(class)


## Find High Correlation

correlationMatrix <- cor(data[,1:55])

high_correlation_variable <- correlationMatrix[,55][correlationMatrix[,55]>0.9]
high_correlation_variable <- high_correlation_variable[-length(high_correlation_variable)]
high_correlation_variable

## Split to train and test

data$Class <- as.factor(data$Class)
set.seed(1)
train = sample(1:nrow(data),85)
data.train = data[train,]
data.test = data[-train,]


## Build lists to store test errors of different methods

KNN_1 <- rep(0,30)
KNN_cv <- rep(0,30)
LDA <- rep(0,30)
QDA <- rep(0,30)
Lasso_full <- rep(0,30)
Lasso_selected <- rep(0,30)
DT <- rep(0,30)
bag <- rep(0,30)
rf_full <- rep(0,30)
rf_selected <- rep(0,30)
boost <- rep(0,30)

## KNN

x3=model.matrix(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data.train)[,-1]
x4=model.matrix(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data.test)[,-1]
y1=data.train$Class

trControl <- trainControl(method  = "cv",
                          number  = 5)

set.seed(1)
fit <- train(Class ~ Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:50),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = data)
fit

set.seed(1)

knn.pred=knn(x3,x4,y1,k=1)
KNN_1[1]<-1-mean(knn.pred==data.test$Class)

knn.pred.cv=knn(x3,x4,y1,k=12)
KNN_cv[1]<-1-mean(knn.pred.cv==data.test$Class)


## LDA

lda.fit=lda(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data=data.train)
lda.fit
lda.pred=predict(lda.fit, newdata=data.test)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class,data.test$Class)
cat('Test error of LDA:',1-mean(lda.class==data.test$Class))
LDA[1]<-1-mean(lda.class==data.test$Class)

## QDA

qda.fit=qda(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data=data.train)
qda.fit
qda.pred=predict(qda.fit, newdata=data.test)
qda.class=qda.pred$class
table(qda.class,data.test$Class)
cat('Test error of QDA:',1-mean(qda.class==data.test$Class))
QDA[1]<-1-mean(qda.class==data.test$Class)

## Lasso with all predictors

x1=model.matrix(Class~.,data.train)[,-1]
y1=data.train$Class
x2=model.matrix(Class~.,data.test)[,-1]

set.seed(1)
cv.out1=cv.glmnet(x1,y1,alpha=1,family="binomial")
bestlam1=cv.out1$lambda.min
plot(cv.out1)

lasso.mod1=glmnet(x1,y1,alpha=1,lambda=bestlam1,family="binomial")
lasso.coef1=coef(lasso.mod1)[,1]
lasso.coef1[lasso.coef1!=0]
summary(lasso.mod1)

pred.lasso1 = predict(lasso.mod1, s = bestlam1, newx = x2, type='class')

cat('Best pruning parameter is ',bestlam1)

cat('.  Test error of Lasso with all predictors:',1-mean(pred.lasso1==data.test$Class))
Lasso_full[1] <- 1-mean(pred.lasso1==data.test$Class)

## Lasso with selected predictors

x3=model.matrix(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data.train)[,-1]
x4=model.matrix(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data.test)[,-1]

set.seed(1)
cv.out2=cv.glmnet(x3,y1,alpha=1,family="binomial")
bestlam2=cv.out2$lambda.min
plot(cv.out2)

lasso.mod2=glmnet(x3,y1,alpha=1,lambda=bestlam2,family="binomial")
lasso.coef2=coef(lasso.mod2)[,1]
lasso.coef2[lasso.coef2!=0]
summary(lasso.mod2)

pred.lasso2 = predict(lasso.mod2, s = bestlam2, newx = x4, type='class')

cat('Best pruning parameter is ',bestlam2)
cat('.  Test error of Lasso with selected predictors:',1-mean(pred.lasso2==data.test$Class))
Lasso_selected[1] <- 1-mean(pred.lasso2==data.test$Class)

## Decision Tree, 'Class' has to be modified to a factor variable for this alg.

data$Class <- as.factor(data$Class)
set.seed(1)
train = sample(1:nrow(data),85)
data.train = data[train,]
data.test = data[-train,]

tree.data = tree(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data=data.train)
summary(tree.data)
plot(tree.data)
text(tree.data)
tree.pred = predict(tree.data,newdata=data.test,type="class")
cat('Test error of Decision Tree:',1-mean(tree.pred==data.test$Class))
DT[1]<-1-mean(tree.pred==data.test$Class)

## Bagging with all predictors

bag.data = randomForest(Class~.,data=data.train,mtry=54,
                        importance = TRUE)
bag.pred=predict(bag.data,newdata=data.test)
cat('Test error of Bagging:',1-mean(bag.pred==data.test$Class))
bag[1]<-1-mean(bag.pred==data.test$Class)

## Random Forest with all predictors

set.seed(1)
rf.data = randomForest(Class~.,data=data.train,mtry=sqrt(54),
                       importance = TRUE)
rf.pred=predict(rf.data,newdata=data.test)
cat('Test error of random forest with all predictors:',1-mean(rf.pred==data.test$Class))
rf_full[1]<-1-mean(rf.pred==data.test$Class)

## Random Forest with selected variables

rf1.data = randomForest(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,
                        data=data.train,mtry=sqrt(8),
                        importance = TRUE)
importance(rf1.data)
rf1.pred=predict(rf1.data,newdata=data.test)
cat('Test error of random forest with selected predictors:',1-mean(rf1.pred==data.test$Class))
rf_selected[1]<-1-mean(rf1.pred==data.test$Class)

## Boosting with selected predictors. Change 'Class' back to numerical variable for this alg.

data <- read.csv("C:/Cornell/4740/final/divorce.csv",header = TRUE)
set.seed(1)
train = sample(1:nrow(data),85)
data.train = data[train,]
data.test = data[-train,]

set.seed(1)
boost.data=gbm(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data=data.train,distribution='bernoulli',n.trees=1000,
               shrinkage = 0.01,cv.folds=10,
               interaction.depth=4)
summary(boost.data)
yhat.boost = predict(boost.data,newdata=data.test,n.trees=1000,type='response')
cat('Test error of boosting:',1-mean(round(yhat.boost)==data.test$Class))
boost[1]<-1-mean(round(yhat.boost)==data.test$Class)

## Repeat random sampling for seed = 2:30

data$Class <- as.factor(data$Class)
for (i in 2:30){
  set.seed(i)
  train = sample(1:nrow(data),85)
  data.train = data[train,]
  data.test = data[-train,]
  x3=model.matrix(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data.train)[,-1]
  x4=model.matrix(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data.test)[,-1]
  y1=data.train$Class
  x1=model.matrix(Class~.,data.train)[,-1]
  x2=model.matrix(Class~.,data.test)[,-1]
  
  set.seed(1)
  
  knn.pred=knn(x3,x4,y1,k=1)
  KNN_1[i]<-1-mean(knn.pred==data.test$Class)
  
  knn.pred.cv=knn(x3,x4,y1,k=12)
  KNN_cv[i]<-1-mean(knn.pred.cv==data.test$Class)
  
  lda.fit=lda(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data=data.train)
  lda.pred=predict(lda.fit, newdata=data.test)
  lda.class=lda.pred$class
  LDA[i]<-1-mean(lda.class==data.test$Class)
  
  qda.fit=qda(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data=data.train)
  qda.pred=predict(qda.fit, newdata=data.test)
  qda.class=qda.pred$class
  QDA[i]<-1-mean(qda.class==data.test$Class)
  
  set.seed(1)
  lasso.mod1=glmnet(x1,y1,alpha=1,lambda=bestlam1,family="binomial")
  lasso.coef1=coef(lasso.mod1)[,1]
  pred.lasso1 = predict(lasso.mod1, s = bestlam1, newx = x2, type='class')
  Lasso_full[i] <- 1-mean(pred.lasso1==data.test$Class)
  
  set.seed(1)
  lasso.mod2=glmnet(x3,y1,alpha=1,lambda=bestlam2,family="binomial")
  lasso.coef2=coef(lasso.mod2)[,1]
  pred.lasso2 = predict(lasso.mod2, s = bestlam2, newx = x4, type='class')
  Lasso_selected[i] <- 1-mean(pred.lasso2==data.test$Class)
  
  tree.data = tree(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,data=data.train)
  tree.pred = predict(tree.data,newdata=data.test,type="class")
  DT[i]<-1-mean(tree.pred==data.test$Class)
  
  bag.data = randomForest(Class~.,data=data.train,mtry=54,
                          importance = TRUE)
  bag.pred=predict(bag.data,newdata=data.test)
  bag[i]<-1-mean(bag.pred==data.test$Class)
  
  set.seed(1)
  rf.data = randomForest(Class~.,data=data.train,mtry=sqrt(54),
                         importance = TRUE)
  rf.pred=predict(rf.data,newdata=data.test)
  rf_full[i]<-1-mean(rf.pred==data.test$Class)
  
  set.seed(1)
  rf1.data = randomForest(Class~Atr9+Atr11+Atr15+Atr17+Atr18+Atr19+Atr20+Atr40,
                          data=data.train,mtry=sqrt(8),
                          importance = TRUE)
  rf1.pred=predict(rf1.data,newdata=data.test)
  rf_selected[i]<-1-mean(rf1.pred==data.test$Class)
}

## Try boosting

data <- read.csv("C:/Cornell/4740/final/divorce.csv",header = TRUE)
for (i in 2:30){
  set.seed(i)
  train = sample(1:nrow(data),85)
  data.train = data[train,]
  data.test = data[-train,]
  
  set.seed(1)
  boost.data=gbm(Class~.,data=data.train,distribution='bernoulli',n.trees=1000,
                 shrinkage = 0.01,cv.folds=10,
                 interaction.depth=4)
  yhat.boost = predict(boost.data,newdata=data.test,n.trees=1000,type='response')
  boost[i]<-1-mean(round(yhat.boost)==data.test$Class)
}

## Boxplot

boxplot(KNN_1,KNN_cv,LDA,QDA,Lasso_full,Lasso_selected,DT,bag,rf_full,rf_selected,boost,
        main = "Test Error of Methods",ylab = "test error",xlab="Methods")