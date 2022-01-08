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
library(pROC)
#From IBM Terminal
df <- read_csv("/home/2021/nyu/fall/ap5254/hw01/NYPD_Complaint_Data_Current__Year_To_Date_.csv")
#If you have the file
df <- read_csv("C:/Users/poona/Downloads/NYPD_Complaint_Data_Current__Year_To_Date_.csv")
#If you have the link
#df = read.csv(url("https://data.cityofnewyork.us/api/views/5uac-w243/rows.csv?accessType=DOWNLOAD"))
#Check dimensions
dim(df)

#Change names to numbers to help reduce bias
names(df) = c(1:36)
names(df)

#Label the dependent variable
names(df)[14] = 'Dependent'
names(df)

head(df)

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

df = df[-c(8,11,14)]
str(df)

#Imbalance check
sum(df$Dependent=='FELONY')/nrow(df)
sum(df$Dependent=='MISDEMEANOR')/nrow(df)
sum(df$Dependent=='VIOLATION')/nrow(df)
#Imbalance check
sum(df$Dependent=='FELONY')
sum(df$Dependent=='MISDEMEANOR')
sum(df$Dependent=='VIOLATION')
dff = df[df[, "Dependent"] == 'FELONY',]
dfm = df[df[, "Dependent"] == 'MISDEMEANOR',]
dfv = df[df[, "Dependent"] == 'VIOLATION',]
dff = dff[sample(nrow(dff), 34552), ]
dfm = dfm[sample(nrow(dfm), 34552), ]
df = rbind(dff,dfm,dfv)

ggplot(data = df) +
  geom_bar(mapping = aes(Dependent))

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

sum(complete.cases(df)=='TRUE')
df=df[complete.cases(df),]

set.seed(222)
index = sample(2, nrow(df), replace = TRUE, prob = c(0.7,0.3))
train = df[index==1,]
test = df[index==2,]
str(train)
summary(train)

train$J = NULL
test$J = NULL

str(df)
#VIF
library(car)
m = lm(A~F+S+T+U+V, data=df)
vif(m)

library(doParallel)

ncores = detectCores(logical = TRUE)
ncores
cl = makePSOCKcluster(5)
registerDoParallel(cl)

start.time = proc.time()
#CrossValidation
library(caTools)
set.seed(223)
train.control <- trainControl(method = "cv", number = 10)
CV <- train(Dependent ~., data = train, method = "LogitBoost", trControl = train.control)
CV
cvpred = predict(CV, test, decision.values = TRUE, type ='raw')
confusionMatrix(cvpred, test$Dependent)
#Bagging
set.seed(224)
library(ipred)
bag = bagging(Dependent ~ ., data = train,nbagg = 150,coob = TRUE)
bagpred = predict(bag, test, decision.values = TRUE, type ='class')
confusionMatrix(as.factor(bagpred$class), test$Dependent)
confusionMatrix(bagpred, test$Dependent)
#Boosting
set.seed(225)
library(gbm)
boost=gbm(Dependent ~ ., data = train,distribution = "multinomial",n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)
summary(boost)
boostpred = predict(boost, test, decision.values = TRUE, type ='response')
boostpred[1:6,,]
p.boostpred = apply(boostpred,1,which.max)
# 1 = Felony, 2 = Misdemeanor, 3 = Violation
head(p.boostpred)
str(p.boostpred)
boostdf = data.frame(p.boostpred)
boostdf$predictions = as.factor(ifelse(boostdf$p.boostpred == 1, "FELONY",
                                ifelse(boostdf$p.boostpred == 2, "MISDEMEANOR","VIOLATION")))
confusionMatrix(boostdf$predictions, test$Dependent) 
stopCluster(cl)
stop.time = proc.time()
run.time = start.time = stop.time
run.time
