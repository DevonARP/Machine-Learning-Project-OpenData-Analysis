#Load the data
library(readr)
library(plyr)
library(dplyr)
library(tidyverse)
library(usethis)
library(devtools)
#install_github("vqv/ggbiplot", force=TRUE)
library(grid)
library(ggbiplot)
library(ggplot2)
library(lattice)
library(caret)
#From IBM Terminal
#df <- read_csv("/home/2021/nyu/fall/ap5254/hw01/NYPD_Complaint_Data_Current__Year_To_Date_.csv")
#If you have the file
#df <- read_csv("C:/Users/poona/Downloads/NYPD_Complaint_Data_Current__Year_To_Date_.csv")
#If you have the link
df = read.csv(url("https://data.cityofnewyork.us/api/views/5uac-w243/rows.csv?accessType=DOWNLOAD"))
#Check dimensions
dim(df)

#Change names to numbers to help reduce bias
names(df) = c(1:36)
names(df)

#Label the dependent variable
names(df)[14] = 'Dependent'
names(df)

head(df)

#Imbalance check
sum(df$Dependent=='FELONY')/nrow(df)
sum(df$Dependent=='MISDEMEANOR')/nrow(df)
sum(df$Dependent=='VIOLATION')/nrow(df)
sum(df$Dependent=='FELONY')
sum(df$Dependent=='MISDEMEANOR')
sum(df$Dependent=='VIOLATION')
dff = df[df[, "Dependent"] == 'FELONY',]
dfm = df[df[, "Dependent"] == 'MISDEMEANOR',]
dfv = df[df[, "Dependent"] == 'VIOLATION',]
dff = dff[sample(nrow(dff), 54797), ]
dfm = dff[sample(nrow(dfm), 54797), ]
dfnew = rbind(dfv,dff,dfm)
dfnew

ggplot(data = df) +
  geom_bar(mapping = aes(Dependent))

#Get rid of Identifier
df = df[-c(1)]

#Data Types of each column
str(df)

#Removing columns that have missing values summing at least half of the total amount of observations
colSums(is.na(df))
nrow(df)/2
df = df[-c(5,6,8,9,16,22,26)]

#Removing columns that are a description of another column
df = df[-c(11,14)]
str(df)

#Principle Component Analysis
pca<- prcomp(df[,c(1,21,22,23,24)], center = TRUE,scale. = TRUE)
str(pca)
#ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#91.2

pca<- prcomp(df[,c(1,21,22)], center = TRUE,scale. = TRUE)
str(pca)
#ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#89.5

pca<- prcomp(df[,c(1,23,24)], center = TRUE,scale. = TRUE)
str(pca)
#ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#89.5

pca<- prcomp(df[,c(1,8)], center = TRUE,scale. = TRUE)
str(pca)
#ggbiplot(pca,labels = df$`13`, ellipse=TRUE, groups=df$Dependent)
#Removing column 8 as that seems to be a 100% correlated to the dependent variable
df = df[-c(8)]
df = df[-c(24,25)]

#randomForest and Variable Importance
library(randomForest)
df$Dependent = as.factor(df$Dependent)
table(df$Dependent)
colnames(df) = c('A','B','C','D','E','F','G','Dependent','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V')

df$B = as.factor(df$B)
df$C = as.factor(df$C)
df$E = as.factor(df$E)
df$G = as.factor(df$G)
df$H = as.factor(df$H)
df$I = as.factor(df$I)
df$K = as.factor(df$K)
df$L = as.factor(df$L)
df$M = as.factor(df$M)
df$N = as.factor(df$N)
df$O = as.factor(df$O)
df$P = as.factor(df$P)
df$Q = as.factor(df$Q)
df$R = as.factor(df$R)

df$C = NULL
df$D = NULL
df$K = NULL
df$L = NULL

set.seed(222)
index = sample(2, nrow(df), replace = TRUE, prob = c(0.7,0.3))
train = df[index==1,]
test = df[index==2,]
str(train)
summary(train)

rf = randomForest(Dependent~., train, na.action=na.omit)
rf

VI =varImp(rf)
varImpPlot(rf,main="Variable Importance")

str(df)
#VIF
library(car)
m = lm(A~F+J+S+T+U+V, data=df)
vif(m)

#Individual Classifiers 
library(pROC)


test= na.omit(test)
#NaiveBayes

#NaiveBayes (No J)
library(e1071)
set.seed(222)
nbtrain = train
nbtest = test
nbtrain$J = NULL
nbtest$J = NULL
nbDefault2 = naiveBayes(Dependent~., data=nbtrain, prob = TRUE)
nbDefault2
nbDefault_pred2 = predict(nbDefault2, nbtest, type="class", prob = TRUE)
nbDefault_pred2
confusionMatrix(nbDefault_pred2, nbtest$Dependent)
plot(nbDefault_pred2)
roc_nb_test2 <- roc(response = nbtest$Dependent, predictor =as.numeric(nbDefault_pred))
plot(roc_nb_test2)
plot(roc_nb_test2,add = TRUE,col = "red", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)


#KNN (No J)
library(class)
knndf = df
str(knndf)
knndf$B = as.numeric(knndf$B)
knndf$E = as.numeric(knndf$E)
knndf$G = as.numeric(knndf$G)
knndf$H = as.numeric(knndf$H)
knndf$I = as.numeric(knndf$I)
knndf$M = as.numeric(knndf$M)
knndf$N = as.numeric(knndf$N)
knndf$O = as.numeric(knndf$O)
knndf$P = as.numeric(knndf$P)
knndf$Q = as.numeric(knndf$Q)
knndf$R = as.numeric(knndf$R)
knntrain2 = knndf[index==1,]
knntest2 = knndf[index==2,]
knntrain2$J = NULL
knntest2$J = NULL
train_knn2 = na.omit(knntrain2)
test_knn2 = na.omit(knntest2)
train_knn_dep2 = train_knn2
test_knn_dep2 = test_knn2
train_knn2$Dependent=NULL
test_knn2$Dependent=NULL

knnModel2 = knn(train = train_knn2, test = test_knn2, cl = train_knn_dep2$Dependent, prob = TRUE)
knnModel2
confusionMatrix(knnModel2, test_knn_dep2$Dependent)
knnroc2=roc(test_knn_dep2$Dependent, attributes(knnModel2)$prob)
plot(knnroc2)
plot(knnroc2,add = TRUE,col = "red", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)


#Decision Tree (No J)
library(rpart)
library(rpart.plot)
dttrain = train
dttrain$J= NULL
dttest = test
dttest$J= NULL
dt2=rpart(Dependent~., data=dttrain, method='class')
rpart.plot(dt2,box.palette = "blue")
dt2
printcp(dt2)
dtpred2 = predict(dt2,dttest,type = 'class')
dtpred2
confusionMatrix(dtpred2, test$Dependent)
dtroc2=roc(test$Dependent, as.numeric(dtpred2))
plot(dtroc2)
plot(dtroc2,add = TRUE,col = "red", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)


#SVM (No J)
str(train)
svmtrain2 = train
svmtest2 = test
svmtrain2$B = NULL
svmtrain2$E = NULL
svmtrain2$G = NULL 
svmtrain2$H = NULL
svmtrain2$I = NULL
svmtrain2$M = NULL
svmtrain2$N = NULL
svmtrain2$O = NULL
svmtrain2$P = NULL
svmtrain2$Q = NULL
svmtrain2$R = NULL
svmtest2$B = NULL
svmtest2$E = NULL
svmtest2$G = NULL 
svmtest2$H = NULL
svmtest2$I = NULL
svmtest2$M = NULL
svmtest2$N = NULL
svmtest2$O = NULL
svmtest2$P = NULL
svmtest2$Q = NULL
svmtest2$R = NULL
svmtrain2$J = NULL
svmtest2$J = NULL
svmfit2 = svm(Dependent ~ ., data = svmtrain2, probability = TRUE, type = "C-classification", kernal = "radial", gamma = 0.1, cost = 1)
svmpred2 <- predict(svmfit2, svmtest2, decision.values = TRUE, probability = TRUE, type ='class')
svmfit2
svmpred2
confusionMatrix(svmpred2, svmtest2$Dependent)

roc_svm_test2 <- roc(response = svmtest2$Dependent, predictor =as.numeric(svmpred2))
plot(roc_svm_test2)
plot(roc_svm_test2,add = TRUE,col = "red", print.auc=TRUE, print.auc.x = 0.5, print.auc.y = 0.3)

#Classification Table
ct = data.frame(dttest$Dependent,nbDefault_pred2,knnModel2,dtpred2,svmpred2)
view(ct)
colnames(ct) <- c("Given", "NaiveBayes","Knn","DecisionTree","SVM")
view(ct)
write.csv(ct,"C:\\Users\\poona\\Desktop\\ct.csv")
